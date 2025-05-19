# quality: gold

import abc
import logging
import uuid
from dataclasses import fields
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Type

import pandas as pd
from bionumpy import AminoAcidEncoding

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet, AminoAcidXEncoding, DNANEncoding
from immuneML.data_model.SequenceSet import Repertoire, build_dynamic_airr_sequence_set_dataclass
from immuneML.data_model.bnp_util import bnp_write_to_file, write_yaml, read_yaml, write_dataset_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset, ElementDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DataImport(metaclass=abc.ABCMeta):

    def __init__(self, params: dict, dataset_name: str):
        self.params = DatasetImportParams.build_object(**params) if isinstance(params, dict) else params
        self.dataset_name = dataset_name

    def import_dataset(self) -> Dataset:

        if self.params.dataset_file is not None:
            assert self.params.dataset_file.is_file(), f"DataImport: dataset_file was specified but not found: {self.params.dataset_file}"
            dataset = self.import_dataset_from_yaml()
        else:
            if self.params.is_repertoire is None and self.params.metadata_file is not None:
                self.params.is_repertoire = True

            if self.params.is_repertoire:
                dataset = self.import_repertoire_dataset()
            elif self.params.paired:
                dataset = self.import_receptor_dataset()
            else:
                dataset = self.import_sequence_dataset()

        dataset.labels['organism'] = self.params.organism

        return dataset

    def import_dataset_from_yaml(self):
        dataset_metadata = read_yaml(self.params.dataset_file)
        if dataset_metadata["dataset_type"] == "RepertoireDataset":
            dataset = self.import_repertoire_dataset()
        elif dataset_metadata["dataset_type"] == "ReceptorDataset":
            dataset = self.import_receptor_dataset()
        elif dataset_metadata["dataset_type"] == "SequenceDataset":
            dataset = self.import_sequence_dataset()
        else:
            raise TypeError(
                f"DataImport: Dataset type '{dataset_metadata['dataset_type']}' not recognized (expected one of: RepertoireDataset, ReceptorDataset, SequenceDataset)")

        return dataset

    def import_repertoire_dataset(self) -> RepertoireDataset:
        if self.params.dataset_file is not None and self.params.dataset_file.is_file():
            imported_dataset_yaml = read_yaml(self.params.dataset_file)

            self.params.path = self.params.dataset_file.parent

            if self.params.metadata_file is None:
                self.params.metadata_file = self.params.dataset_file.parent / imported_dataset_yaml['metadata_file']

            imported_identifier = imported_dataset_yaml['identifier']
            imported_labels = imported_dataset_yaml['labels']

            # type_dict = imported_dataset_yaml['type_dict']
        else:
            imported_labels = None
            imported_identifier = None

        metadata = self.import_repertoire_metadata(self.params.metadata_file)
        repertoires = self.load_repertoires(metadata)
        new_metadata_file = ImportHelper.make_new_metadata_file(repertoires, metadata, self.params.result_path,
                                                                self.dataset_name)
        # type_dict = self.determine_repertoire_type_dict(repertoires)
        labels = self.determine_repertoire_dataset_labels(metadata, imported_labels=imported_labels)

        dataset_filename = self.params.result_path / f"{self.dataset_name}.yaml"
        dataset_yaml = RepertoireDataset.create_metadata_dict(metadata_file=new_metadata_file,
                                                              # type_dict=type_dict,
                                                              labels=labels,
                                                              name=self.dataset_name,
                                                              identifier=imported_identifier)
        write_dataset_yaml(dataset_filename, dataset_yaml)

        return RepertoireDataset(labels=labels,
                                 repertoires=repertoires, metadata_file=new_metadata_file, name=self.dataset_name,
                                 dataset_file=dataset_filename, identifier=dataset_yaml["identifier"])

    def import_repertoire_metadata(self, metadata_file_path):
        try:
            metadata = pd.read_csv(metadata_file_path, sep=",")
            if "identifier" in metadata.columns:
                assert len(list(metadata["identifier"])) == len(set(list(metadata["identifier"]))), \
                    (f"DataImport: if the field 'identifier' is supplied, each repertoire must have "
                     f"a unique identifier (found {len(set(list(metadata['identifier'])))} unique "
                     f"identifiers for {len(list(metadata['identifier']))} repertoires).")
        except Exception as e:
            raise Exception(f"{e}\nAn error occurred while reading in the metadata file {self.params.metadata_file}. "
                            f"Please see the error log above for more details on this error and the documentation "
                            f"for the expected format of the metadata.").with_traceback(e.__traceback__)

        ParameterValidator.assert_keys_present(metadata.columns.tolist(), ["filename"], self.__class__.__name__,
                                               f'{self.dataset_name}: params: metadata_file')
        return metadata

    def load_repertoires(self, metadata):
        PathBuilder.build(self.params.result_path / "repertoires/")

        with Pool(self.params.number_of_processes) as pool:
            repertoires = pool.map(self.load_repertoire_object, [row for _, row in metadata.iterrows()])

        return repertoires

    def determine_repertoire_type_dict(self, repertoires):
        try:
            if all(repertoires[0].metadata['type_dict_dynamic_fields'] == rep.metadata['type_dict_dynamic_fields'] for
                   rep
                   in repertoires[1:]):
                return repertoires[0].metadata['type_dict_dynamic_fields']
            else:
                raise RuntimeError()
        except Exception as e:
            logging.warning(f'{DataImport.__name__}: dynamic fields for the dataset {self.dataset_name} could not be '
                            f'extracted, some repertoires have different fields.')
            return {}

    def determine_repertoire_dataset_labels(self, metadata, imported_labels=None):
        potential_label_names = list(
            set(metadata.columns.tolist()) - {"filename", "type_dict_dynamic_fields", "identifier", "subject_id"})
        potential_labels = {key: list(set(metadata[key].values.tolist())) for key in potential_label_names}

        if imported_labels is not None:
            labels = imported_labels
            if any(label not in potential_label_names for label in imported_labels):
                logging.warning(f"{DataImport.__name__}: an error occurred when importing dataset {self.dataset_name}. "
                                f"Labels specified in the dataset file ({imported_labels}) could not be found in the repertoire "
                                f"fields. Proceeding with the following labels: {potential_labels}.")
                labels = potential_labels
        else:
            labels = potential_labels

        return labels

    def import_element_dataset(self, dataset_class: Type, filter_func=None):
        if self.params.dataset_file is not None and self.params.dataset_file.is_file():
            filenames = [Path(read_yaml(self.params.dataset_file)["filename"])]
            if self.params.path.is_file():
                filenames[0] = self.params.path
            elif self.params.path:
                filenames[0] = self.params.path / Path(filenames[0]).name
            else:
                filenames[0] = self.params.dataset_file.parent / Path(filenames[0]).name
            assert filenames[0].is_file(), \
                f"DataImport: filename {filenames[0]} specified in dataset file was not found."
        else:
            filenames = ImportHelper.get_sequence_filenames(self.params.path, self.dataset_name)

        final_data_dict = self._construct_element_dataset_data_dict(filenames, filter_func)

        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(final_data_dict)
        filename = self._write_element_dataset_tsv_file(final_data_dict, bnp_dc)

        possible_labels = {key: list(set(final_data_dict[key])) for key in type_dict.keys()}
        logging.info(f"All possible labels for dataset '{self.dataset_name}' are: {list(possible_labels.keys())}")

        if self.params.label_columns:
            label_variants = self.params.label_columns + [ImportHelper.get_standardized_name(label_name) for label_name
                                                          in self.params.label_columns]
            possible_labels = {key: value for key, value in possible_labels.items() if key in label_variants}

        dataset_filename = self._write_element_dataset_metadata_file(dataset_class, filename, type_dict,
                                                                     possible_labels)
        metadata = read_yaml(dataset_filename)

        logging.info(f"Displayed labels for dataset '{self.dataset_name}' are: {list(metadata['labels'].keys())}")

        return dataset_class(name=self.dataset_name, bnp_dataclass=bnp_dc, dataset_file=dataset_filename,
                             dynamic_fields=type_dict, filename=filename, labels=metadata['labels'])

    def import_sequence_dataset(self) -> SequenceDataset:
        return self.import_element_dataset(SequenceDataset)

    def import_receptor_dataset(self) -> ReceptorDataset:
        return self.import_element_dataset(ReceptorDataset,
                                           lambda df: ImportHelper.filter_illegal_receptors(df, self.params.receptor_chains))

    def _construct_element_dataset_data_dict(self, filenames, filter_func) -> dict:
        final_df = None

        for filename in filenames:
            df = self.load_sequence_dataframe(filename)
            if filter_func:
                df = filter_func(df)
            final_df = pd.concat([final_df, df])

        return final_df.to_dict(orient='list')

    def _write_element_dataset_tsv_file(self, final_data_dict, bnp_dc) -> Path:
        PathBuilder.build(self.params.result_path)

        data = bnp_dc(**final_data_dict)
        data_filename = self.params.result_path / f'{self.dataset_name}.tsv'
        bnp_write_to_file(data_filename, data)

        return data_filename

    def _write_element_dataset_metadata_file(self, dataset_class, filename, type_dict, possible_labels):
        dataset_filename = self.params.result_path / f"{self.dataset_name}.yaml"

        if self.params.dataset_file:
            metadata = read_yaml(self.params.dataset_file)
        else:
            metadata = ElementDataset.create_metadata_dict(dataset_class=dataset_class.__name__,
                                                           filename=filename,
                                                           type_dict=type_dict,
                                                           name=self.dataset_name,
                                                           labels=possible_labels)
        write_dataset_yaml(dataset_filename, metadata)
        return dataset_filename

    def load_repertoire_object(self, metadata_row: pd.Series) -> Repertoire:
        try:
            filename = ImportHelper.get_repertoire_filename_from_metadata_row(metadata_row, self.params)

            dataframe = self.load_sequence_dataframe(filename)

            repertoire_inputs = {**{"metadata": metadata_row.to_dict(),
                                    "path": self.params.result_path / "repertoires/",
                                    "filename_base": filename.stem}, **dataframe.to_dict(orient='list')}

            if "identifier" in metadata_row:
                repertoire_inputs["identifier"] = metadata_row["identifier"]

            repertoire = Repertoire.build(**repertoire_inputs)

            if repertoire.get_element_count() == 0:
                logging.warning(
                    f"Repertoire {repertoire.identifier} contains 0 sequences. It is recommended to remove this repertoire from the dataset. ")

            return repertoire
        except Exception as e:
            raise RuntimeError(
                f"{self.__class__.__name__}: error when importing file {Path(metadata_row['filename']).absolute()}: "
                f"\n{e}.\n")

    def load_sequence_dataframe(self, filename: Path):

        df = self.load_file(filename)
        df = self.preprocess_file(df)

        df = ImportHelper.standardize_column_names(df)
        df = ImportHelper.add_cdr3_from_junction(df)
        df = ImportHelper.drop_empty_sequences(df, self.params.import_empty_aa_sequences,
                                               self.params.import_empty_nt_sequences,
                                               self.params.region_type)
        df = ImportHelper.drop_illegal_character_sequences(df, self.params.import_illegal_characters,
                                                           self.params.import_with_stop_codon, self.params.region_type)
        df = ImportHelper.filter_illegal_sequences(df, self.params, self.dataset_name)
        df = ImportHelper.extract_locus_from_data(df, self.params, self.dataset_name)
        df = ImportHelper.add_default_fields_for_airr_seq_set(df)

        assert df.columns.shape[0] == df.columns.unique().shape[0], \
            f"There are non-unique columns in the imported file {filename.name}: {df.columns.tolist()}"

        return df

    def preprocess_file(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sequence_id' in df.columns:
            missing_seq_ids = df['sequence_id'].isnull() | df['sequence_id'].eq('')
            df.loc[missing_seq_ids, 'sequence_id'] = [uuid.uuid4().hex for _ in range(missing_seq_ids.sum())]
        return df

    def load_file(self, filename: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(str(filename), sep=self.params.separator, iterator=False,
                             usecols=self.params.columns_to_load)
        except ValueError:

            df = pd.read_csv(str(filename), sep=self.params.separator, iterator=False)

            expected = [e for e in self.params.columns_to_load if e not in list(df.columns)]
            df = df[[col for col in self.params.columns_to_load if col in df.columns]]

            logging.warning(
                f"{self.__class__.__name__}: expected to find the following column(s) in the input file "
                f"'{filename.name}', which were not found: {expected}. The following columns were imported instead: "
                f"{list(df.columns)}. \nTo remove this warning, add the relevant columns to the input file, "
                f"or change which columns are imported under 'datasets/{self.dataset_name}/params/columns_to_load' and "
                f"'datasets/{self.dataset_name}/params/column_mapping'.")

        if hasattr(self.params, "column_mapping") and self.params.column_mapping is not None:
            df.rename(columns=self.params.column_mapping, inplace=True)

        df = df.dropna(how='all')  # remove all fully empty rows

        df = ImportHelper.standardize_bool_values(df)
        df = ImportHelper.standardize_none_values(df)

        df = self._convert_types(df)

        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = [f.name for f in fields(AIRRSequenceSet)
                    if f.type in [str, AminoAcidEncoding, AminoAcidXEncoding, DNANEncoding] and f.name in df.columns]

        df = df.astype({col: str for col in str_cols})
        df.loc[:, str_cols] = df.loc[:, str_cols].replace('nan', '').replace('-1.0', '')

        encoded_cols = [f for f, t in AIRRSequenceSet.get_field_type_dict().items()
                        if t in [AminoAcidXEncoding, AminoAcidEncoding, DNANEncoding] and f in df.columns]

        df.loc[:, encoded_cols] = df.loc[:, encoded_cols].apply(lambda x: x.str.upper())

        invalid_cols = df.columns[~df.map(type).nunique().eq(1)]
        df[invalid_cols] = df[invalid_cols].astype(str)

        int_cols = [f.name for f in fields(AIRRSequenceSet) if f.type == int and f.name in df.columns]
        df[int_cols] = df[int_cols].astype(int)

        return df
