# quality: gold

import abc
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceSet import Repertoire, build_dynamic_airr_sequence_set_dataclass
from immuneML.data_model.bnp_util import bnp_write_to_file, write_yaml
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

        return dataset

    def import_repertoire_dataset(self) -> RepertoireDataset:
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
        return RepertoireDataset(labels={key: list(set(metadata[key].values.tolist())) for key in potential_labels},
                                 repertoires=repertoires, metadata_file=new_metadata_file, name=self.dataset_name)

    def import_sequence_dataset(self, params: DatasetImportParams, dataset_name: str) -> SequenceDataset:
        filenames = ImportHelper.get_sequence_filenames(params.path, dataset_name)
        final_df = None

        for filename in filenames:
            df = self.load_sequence_dataframe(filename)
            final_df = pd.concat([final_df, df])

        final_data_dict = final_df.to_dict()

        dc, types = build_dynamic_airr_sequence_set_dataclass(final_data_dict)
        filename, dataset_file = self._prepare_values_for_element_dataset(final_data_dict, dc, types)

        return SequenceDataset(name=dataset_name, bnp_dataclass=dc, dataset_file=dataset_file,
                               dynamic_fields=list(types.keys()), filename=filename)

    def import_receptor_dataset(self, params: DatasetImportParams, dataset_name: str) -> ReceptorDataset:
        filenames = ImportHelper.get_sequence_filenames(params.path, dataset_name)
        final_df = None

        for filename in filenames:
            df = self.load_sequence_dataframe(filename)
            df = ImportHelper.filter_illegal_receptors(df)
            final_df = pd.concat([final_df, df])

        final_data_dict = final_df.to_dict()

        dc, types = build_dynamic_airr_sequence_set_dataclass(final_data_dict)

        filename, dataset_file = self._prepare_values_for_element_dataset(final_data_dict, dc, types)

        return ReceptorDataset(name=dataset_name, bnp_dataclass=dc, dataset_file=dataset_file,
                               dynamic_fields=list(types.keys()), filename=filename)


    def _prepare_values_for_element_dataset(self, final_data_dict, dc, types):
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
                    f"{ImportHelper.__name__}: error when importing file {metadata_row['filename']}: {e}")

    def load_sequence_dataframe(self, filename: Path):
        try:
            df = pd.read_csv(str(filename), sep=self.params.separator, iterator=False,
                             usecols=self.params.columns_to_load)
        except ValueError:
            try:
                df = pd.read_csv(str(filename), sep=self.params.separator, iterator=False,
                                 usecols=self.params.columns_to_load)
            except ValueError:
                df = pd.read_csv(str(filename), sep=self.params.separator, iterator=False)

                expected = [e for e in self.params.columns_to_load if e not in list(df.columns)]

                logging.warning(
                    f"ImportHelper: expected to find the following column(s) in the input file '{filename.name}', "
                    f"which were not found: {expected}. The following columns were imported instead: "
                    f"{list(df.columns)}. \nTo remove this warning, add the relevant columns to the input file, "
                    f"or change which columns are imported under 'datasets/<dataset_key>/params/columns_to_load' and "
                    f"'datasets/<dataset_key>/params/column_mapping'.")

        df = ImportHelper.parse_sequence_dataframe(df, self.params, self.dataset_name)

        return df
