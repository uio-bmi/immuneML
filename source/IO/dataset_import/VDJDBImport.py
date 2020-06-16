import pickle
from glob import glob

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.util.ImportHelper import ImportHelper


class VDJDBImport(DataImport):
    """
    Imports data from VDJdb format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter or
    to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences).

    Arguments:

        metadata_file: path to the metadata file, used only when importing repertoires, for receptor datasets,
            metadata information is located together with the sequence data so there is no need for additional file

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vdjdb_dataset:
            format: VDJDB
            params:
                # these parameters have to be always specified:
                metadata_file: path/to/metadata.csv # csv file with fields filename, donor and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                # the following parameter have these default values so these need to be specified only if a different behavior is required
                paired: True # whether to import_dataset paired data: if true returns ReceptorDataset, and if false returns SequenceDataset
                file_size: 1000 # number of sequences / receptors per file as stored internally by ImmuneML in ImmuneML format - not visible to users
                column_mapping:
                    V: v_genes
                    J: j_genes
                    CDR3: sequence_aas
                    complex.id: sequence_identifiers
                region_type: CDR3
                separator: "\\t"

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        vdjdb_params = DatasetImportParams.build_object(**params)
        if vdjdb_params.metadata_file is not None:
            dataset = VDJDBImport.load_repertoire_dataset(vdjdb_params)
        else:
            dataset = VDJDBImport.load_sequence_dataset(vdjdb_params)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams) -> Dataset:
        return ImportHelper.import_repertoire_dataset(VDJDBImport.preprocess_repertoire, params)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        return ImportHelper.load_repertoire_as_dataframe(metadata, params)

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams) -> Dataset:

        filenames = glob(params.path + "*.tsv")
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = VDJdbSequenceImport.import_items(filename, paired=params.paired)

            while len(items) > params.file_size or (index == len(filenames)-1 and len(items) > 0):
                dataset_filenames.append(params.result_path + "batch_{}.pickle".format(file_index))
                VDJDBImport.store_items(dataset_filenames, items, params.file_size)
                items = items[params.file_size:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=params.file_size) if params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=params.file_size)

    @staticmethod
    def store_items(dataset_filenames: list, items: list, file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:file_size], file)
