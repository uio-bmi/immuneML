# quality: gold

import abc
import logging
from dataclasses import fields
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd
from bionumpy import AminoAcidEncoding, DNAEncoding

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet, AminoAcidXEncoding
from immuneML.data_model.SequenceSet import Repertoire, build_dynamic_airr_sequence_set_dataclass
from immuneML.data_model.bnp_util import bnp_write_to_file, write_yaml, read_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DataImport(metaclass=abc.ABCMeta):

    def __init__(self, params: dict, dataset_name: str):
        self.params = DatasetImportParams.build_object(**params)
        self.dataset_name = dataset_name

    def import_dataset(self) -> Dataset:

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

    def import_repertoire_dataset(self) -> RepertoireDataset:

        self.check_or_discover_metadata_file()

        try:
            metadata = pd.read_csv(self.params.metadata_file, sep=",")
        except Exception as e:
            raise Exception(f"{e}\nAn error occurred while reading in the metadata file {self.params.metadata_file}. "
                            f"Please see the error log above for more details on this error and the documentation "
                            f"for the expected format of the metadata.").with_traceback(e.__traceback__)

        ParameterValidator.assert_keys_present(metadata.columns.tolist(), ["filename"], self.__class__.__name__,
                                               f'{self.dataset_name}: params: metadata_file')

        PathBuilder.build(self.params.result_path / "repertoires/")

        with Pool(self.params.number_of_processes) as pool:
            repertoires = pool.map(self.load_repertoire_object, [row for _, row in metadata.iterrows()])

        new_metadata_file = ImportHelper.make_new_metadata_file(repertoires, metadata, self.params.result_path,
                                                                self.dataset_name)

        potential_labels = list(set(metadata.columns.tolist()) - {"filename"})
        dataset_filename = self._make_dataset_file_for_repertoire_dataset(repertoires)

        return RepertoireDataset(labels={key: list(set(metadata[key].values.tolist())) for key in potential_labels},
                                 repertoires=repertoires, metadata_file=new_metadata_file, name=self.dataset_name,
                                 dataset_file=dataset_filename)

    def import_sequence_dataset(self) -> SequenceDataset:
        filenames = ImportHelper.get_sequence_filenames(self.params.path, self.dataset_name)
        final_df = None

        for filename in filenames:
            df = self.load_sequence_dataframe(filename)
            final_df = pd.concat([final_df, df])

        final_data_dict = final_df.to_dict(orient='list')

        dc, types = build_dynamic_airr_sequence_set_dataclass(final_data_dict)
        filename, dataset_file = self._prepare_values_for_element_dataset(final_data_dict, dc, types)

        return SequenceDataset(name=self.dataset_name, bnp_dataclass=dc, dataset_file=dataset_file,
                               dynamic_fields=list(types.keys()), filename=filename)

    def import_receptor_dataset(self) -> ReceptorDataset:
        filenames = ImportHelper.get_sequence_filenames(self.params.path, self.dataset_name)
        final_df = None

        for filename in filenames:
            df = self.load_sequence_dataframe(filename)
            df = ImportHelper.filter_illegal_receptors(df)
            final_df = pd.concat([final_df, df])

        final_data_dict = final_df.to_dict(orient='list')

        dc, types = build_dynamic_airr_sequence_set_dataclass(final_data_dict)

        filename, dataset_file = self._prepare_values_for_element_dataset(final_data_dict, dc, types)

        return ReceptorDataset(name=self.dataset_name, bnp_dataclass=dc, dataset_file=dataset_file,
                               dynamic_fields=list(types.keys()), filename=filename)

    def check_or_discover_metadata_file(self):
        if self.params.metadata_file is None and self.params.dataset_file and self.params.dataset_file.is_file():
            dataset_metadata = read_yaml(self.params.dataset_file)
            if 'metadata_file' in dataset_metadata:
                self.params.metadata_file = self.params.dataset_file.parent / dataset_metadata['metadata_file']

    def _make_dataset_file_for_repertoire_dataset(self, repertoires: List[Repertoire]):
        dataset_filename = self.params.result_path / f"{self.dataset_name}_dataset.yaml"
        metadata = {'dataset_name': self.dataset_name, 'example_count': len(repertoires)}

        try:
            if all(repertoires[0].metadata['type_dict_dynamic_fields'] == rep.metadata['type_dict_dynamic_fields'] for
                   rep
                   in repertoires[1:]):
                metadata['type_dict_dynamic_fields'] = repertoires[0].metadata['type_dict_dynamic_fields']
            else:
                raise RuntimeError()
        except Exception as e:
            logging.warning(f'{DataImport.__name__}: dynamic fields for the dataset {self.dataset_name} could not be '
                            f'extracted, some repertoires have different fields.')

        write_yaml(dataset_filename, metadata)

        return dataset_filename

    def _prepare_values_for_element_dataset(self, final_data_dict, dc, types):
        PathBuilder.build(self.params.result_path)

        data = dc(**final_data_dict)
        bnp_write_to_file(self.params.result_path / f'{self.dataset_name}.tsv', data)

        dataset_filename = self.params.result_path / f"{self.dataset_name}_dataset.yaml"
        metadata = {'type_dict_dynamic_fields': {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in types.items()}}
        write_yaml(dataset_filename, metadata)

        return self.params.result_path / f'{self.dataset_name}.tsv', dataset_filename

    def load_repertoire_object(self, metadata_row: pd.Series) -> Repertoire:
        try:
            filename = ImportHelper.get_repertoire_filename_from_metadata_row(metadata_row, self.params)

            dataframe = self.load_sequence_dataframe(filename)

            repertoire_inputs = {**{"metadata": metadata_row.to_dict(),
                                    "path": self.params.result_path / "repertoires/",
                                    "filename_base": filename.stem}, **dataframe.to_dict(orient='list')}
            repertoire = Repertoire.build(**repertoire_inputs)

            return repertoire
        except Exception as e:
            raise RuntimeError(
                f"{self.__class__.__name__}: error when importing file {metadata_row['filename']}: {e}")

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

        df = ImportHelper.standardize_none_values(df)
        df = self._convert_types(df)

        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = [f.name for f in fields(AIRRSequenceSet)
                    if f.type in [str, AminoAcidEncoding, AminoAcidXEncoding, DNAEncoding] and f.name in df.columns]

        df.loc[:, str_cols] = df.loc[:, str_cols].astype(str).replace('nan', '').replace('-1.0', '')

        invalid_cols = df.columns[~df.map(type).nunique().eq(1)]
        df[invalid_cols] = df[invalid_cols].astype(str)

        int_cols = [f.name for f in fields(AIRRSequenceSet) if f.type == int and f.name in df.columns]
        df[int_cols] = df[int_cols].astype(int)

        return df
