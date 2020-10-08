import airr
import pandas as pd
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
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
        return ImportHelper.import_dataset(AIRRImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        """
        Function for preprocessing data from a dataframe containing AIRR data, such that:
            - productive sequences, sequences with stop codons or out of frame sequences are filtered according to specification
            - if RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the CDR3 definition
            - if no chain column was specified, the chain is extracted from the v gene name
            - the allele information is removed from the V and J genes
        """
        df["frame_types"] = SequenceFrameType.OUT.name

        df.loc[df["productive"], "frame_types"] = SequenceFrameType.IN.name
        if "vj_in_frame" in df.columns:
            df.loc[df["vj_in_frame"], "frame_types"] = SequenceFrameType.IN.name
        if "stop_codon" in df.columns:
            df.loc[df["stop_codon"], "frame_types"] = SequenceFrameType.STOP.name

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        df = df[df["frame_types"].isin(frame_type_list)]

        ImportHelper.junction_to_cdr3(df, params.region_type)

        if "chains" not in df.columns:
            df["chains"] = ImportHelper.load_chains_from_genes(df, "v_genes")

        df["v_genes"] = ImportHelper.strip_alleles(df, "v_genes")
        df["j_genes"] = ImportHelper.strip_alleles(df, "j_genes")

        return df


    @staticmethod
    def alternative_load_func(filename, params):
        return airr.load_rearrangement(filename)

