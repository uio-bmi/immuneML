import pickle

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.util.ImportHelper import ImportHelper


class VDJdbImport(DataImport):
    """
    Imports data from VDJdb format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter or
    to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences).

    Arguments:

        metadata_file: path to the metadata file, used only when importing repertoires, for receptor datasets,
            metadata information is located together with the sequence data so there is no need for additional file

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vdjdb_dataset:
            format: VDJdb
            params:
                # these parameters have to be always specified:
                metadata_file: path/to/metadata.csv # csv file with fields filename, subject_id and arbitrary others which can be used as labels in analysis
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
            dataset = VDJdbImport.load_repertoire_dataset(vdjdb_params, dataset_name)
        else:
            dataset = VDJdbImport.load_sequence_dataset(vdjdb_params, dataset_name)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_repertoire_dataset(VDJdbImport.preprocess_repertoire, params, dataset_name)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        return ImportHelper.load_repertoire_as_dataframe(metadata, params)

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_sequence_dataset(VDJdbSequenceImport.import_items, params, dataset_name, paired=params.paired)

    @staticmethod
    def store_items(dataset_filenames: list, items: list, file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:file_size], file)
