import airr
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.util.ImportHelper import ImportHelper
from source.IO.sequence_import.AIRRSequenceImport import AIRRSequenceImport


class AIRRImport(DataImport):
    """
    Imports the data from an AIRR-formatted .tsv files into a RepertoireDataset.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_airr_dataset:
            format: AIRR
            params:
                # required parameters:
                metadata_file: path/to/metadata.csv
                path: path/to/directory/with/repertoire/files/
                result_path: path/where/to/store/imported/repertoires/
                # optional parameters (if not specified the values bellow will be used):
                import_productive: True # whether to import productive sequences or not to import them
                import_with_stop_codon: False # whether to import sequences with stop codon
                import_out_of_frame: False # whether to import sequences which are out of frame (where vj_in_frame is False)
                columns_to_load: [sequence_aa, sequence, v_call, j_call, locus, duplicate_count, productive, vj_in_frame, stop_codon] # to import other columns, add them to this list
                column_mapping: # AIRR column names -> immuneML repertoire fields
                    junction: sequences
                    junction_aa: sequence_aas
                    v_call: v_genes
                    j_call: j_genes
                    locus: chains
                    duplicate_count: counts
                    sequence_id: sequence_identifiers
                batch_size: 4
                separator: "\\t"
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        airr_params = DatasetImportParams.build_object(**params)
        if airr_params.metadata_file is not None:
            dataset = AIRRImport.load_repertoire_dataset(airr_params, dataset_name)
        else:
            dataset = AIRRImport.load_sequence_dataset(airr_params, dataset_name)
        return dataset


    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_repertoire_dataset(AIRRImport.preprocess_repertoire, params, dataset_name)

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams, dataset_name: str) -> Dataset:
        return ImportHelper.import_sequence_dataset(AIRRSequenceImport.import_items, params, dataset_name,
                                                    import_productive=params.import_productive, import_with_stop_codon=params.import_with_stop_codon,
                                                    import_out_of_frame=params.import_out_of_frame, region_type=params.region_type,
                                                    region_definition=params.region_definition, column_mapping=params.column_mapping,
                                                    paired=params.paired)


    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams):

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params,
                                                       alternative_load_func=AIRRImport._load_rearrangement_wrapper)

        df = AIRRSequenceImport.preprocess_sequence_dataframe(df, vars(params))
        return df


    @staticmethod
    def _load_rearrangement_wrapper(filename, params):
        return airr.load_rearrangement(filename)

