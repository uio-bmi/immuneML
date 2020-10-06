import airr
from pandas import DataFrame

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.util.ImportHelper import ImportHelper


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

        dataset = ImportHelper.load_dataset_if_exists(params, airr_params, dataset_name)

        if dataset is None:
            if airr_params.is_repertoire:
                dataset = ImportHelper.import_repertoire_dataset(AIRRImport.preprocess_repertoire, airr_params, dataset_name)
            else:
                dataset = ImportHelper.import_sequence_dataset(AIRRImport.import_all_sequences, airr_params, dataset_name)

        return dataset


    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams):
        df = ImportHelper.load_repertoire_as_dataframe(metadata, params, alternative_load_func=AIRRImport._load_rearrangement_wrapper)
        df = AIRRImport.preprocess_dataframe(df, params)
        return df


    @staticmethod
    def preprocess_dataframe(df: DataFrame, params: DatasetImportParams):
        """
        Function for preprocessing data from a dataframe containing AIRR data, such that:
            - productive sequences, sequences with stop codons or out of frame sequences are filtered according to specification
            - if RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the CDR3 definition
            - if no chain column was specified, the chain is extracted from the v gene name
            - the allele information is removed from the V and J genes
        """
        if params.import_with_stop_codon is False and "stop_codon" in df.columns:  # todo move to importhelper? this is general??
            df = df[~df["stop_codon"]]
        if params.import_out_of_frame is False and "vj_in_frame" in df.columns:
            df = df[df["vj_in_frame"]]
        if params.import_productive is False and "productive" in df.columns:
            df = df[~df["productive"]]
        if params.import_with_stop_codon is False and params.import_out_of_frame is False:
            df = df[df["productive"]]

        ImportHelper.junction_to_cdr3(df, params.region_definition, params.region_type)

        if "chains" not in df.columns:
            df["chains"] = ImportHelper.load_chains_from_genes(df, "v_genes")

        df["v_genes"] = ImportHelper.strip_alleles(df, "v_genes")
        df["j_genes"] = ImportHelper.strip_alleles(df, "j_genes")

        return df


    @staticmethod
    def import_all_sequences(path, params):
        df = ImportHelper.load_sequence_dataframe(path, params, alternative_load_func=AIRRImport._load_rearrangement_wrapper)
        df = AIRRImport.preprocess_dataframe(df, params)

        if params.paired:
            raise NotImplementedError("AIRRImport: import of paired receptor AIRR data has not been implemented.")
        else:
            sequences = df.apply(ImportHelper.import_sequence, axis=1).values

        return sequences


    @staticmethod
    def _load_rearrangement_wrapper(filename, params):
        return airr.load_rearrangement(filename)

