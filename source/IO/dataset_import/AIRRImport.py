import airr
import pandas as pd

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.ImportHelper import ImportHelper


class AIRRImport(DataImport):
    """
    Imports data in AIRR format into a Repertoire-, or SequenceDataset.
    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
    antigen specificity.

    AIRR rearrangement schema can be found here: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

    When importing a ReceptorDataset, the airr field cell_id is used to determine the chain pairs.

    Arguments:

        path (str): This is the path to a directory with AIRR files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset.
        By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        Only the AIRR files included under the column 'filename' are imported into the RepertoireDataset.
        For setting SequenceDataset metadata, metadata_file is ignored, see metadata_column_mapping instead.

        import_productive (bool): Whether productive sequences (with value 'T' in column productive) should be included
        in the imported sequences. By default, import_productive is True.

        import_with_stop_codon (bool): Whether sequences with stop codons (with value 'T' in column stop_codon) should
        be included in the imported sequences. This only applies if column stop_codon is present. By default,
        import_with_stop_codon is False.

        import_out_of_frame (bool): Whether out of frame sequences (with value 'F' in column vj_in_frame) should
        be included in the imported sequences. This only applies if column vj_in_frame is present. By default,
        import_out_of_frame is False.

        region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the
        first and last amino acids are removed from the CDR3 sequence, as AIRR uses the IMGT junction. Specifying
        any other value will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~source.data_model.receptor.RegionType.RegionType` enum.

        column_mapping (dict): A mapping from AIRR column names to immuneML's internal data representation.
        For AIRR, this is by default set to:

        .. indent with spaces
        .. code-block:: yaml

                junction: sequences
                junction_aa: sequence_aas
                v_call: v_genes
                j_call: j_genes
                locus: chains
                duplicate_count: counts
                sequence_id: sequence_identifiers

        A custom column mapping can be specified here if necessary (for example; adding additional data fields if
        they are present in the AIRR file, or using alternative column names).
        Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS

        metadata_column_mapping (dict): Specifies metadata for SequenceDatasets. This should specify a mapping similar
        to column_mapping where keys are AIRR column names and values are the names that are internally used in immuneML
        as metadata fields. These metadata fields can be used as prediction labels for SequenceDatasets.
        For AIRR format, there is no default metadata_column_mapping.
        For setting RepertoireDataset metadata, metadata_column_mapping is ignored, see metadata_file instead.

        separator (str): Column separator, for AIRR this is by default "\\t".

        import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False

        import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on
        amino acid sequences, this parameter will typically be False (import only non-empty amino acid sequences)


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_airr_dataset:
            format: AIRR
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                metadata_column_mapping: # metadata column mapping AIRR: immuneML for SequenceDataset
                    airr_column_name1: metadata_label1
                    airr_column_name2: metadata_label2
                import_productive: True # whether to include productive sequences in the dataset
                import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                import_out_of_frame: False # whether to include out of frame sequences in the dataset
                # Optional fields with AIRR-specific defaults, only change when different behavior is required:
                separator: "\\t" # column separator
                region_type: IMGT_CDR3 # what part of the sequence to import
                column_mapping: # column mapping AIRR: immuneML
                    junction: sequences
                    junction_aa: sequence_aas
                    v_call: v_genes
                    j_call: j_genes
                    locus: chains
                    duplicate_count: counts
                    sequence_id: sequence_identifiers
                import_empty_nt_sequences: True # keep sequences even if the `sequences` column is empty (provided that other fields are as specified here)
                import_empty_aa_sequences: False # remove all sequences with empty `sequence_aas` column

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
        if "productive" in df.columns:
            df["frame_types"] = SequenceFrameType.OUT.name
            df.loc[df["productive"] == True, "frame_types"] = SequenceFrameType.IN.name
        else:
            df["frame_types"] = None

        if "vj_in_frame" in df.columns:
            df.loc[df["vj_in_frame"] == True, "frame_types"] = SequenceFrameType.IN.name
        if "stop_codon" in df.columns:
            df.loc[df["stop_codon"] == True, "frame_types"] = SequenceFrameType.STOP.name

        if "productive" in df.columns:
            frame_type_list = ImportHelper.prepare_frame_type_list(params)
            df = df[df["frame_types"].isin(frame_type_list)]

        if params.region_type == RegionType.IMGT_CDR3:
            if "sequence_aas" not in df.columns and "sequences" not in df.columns:
                if "cdr3" in df.columns:
                    df.rename(columns={"cdr3": "sequences"}, inplace=True)
                if "cdr3_aa" in df.columns:
                    df.rename(columns={"cdr3_aa": "sequence_aas"}, inplace=True)
                df.loc[:, "region_types"] = params.region_type.name
            elif "junction" in params.column_mapping or "junction_aa" in params.column_mapping:
                ImportHelper.junction_to_cdr3(df, params.region_type)
        # todo else: support "full_sequence" import through regiontype?

        if "chains" not in df.columns:
            df.loc[:, "chains"] = ImportHelper.load_chains_from_genes(df, "v_genes")

        df.loc[:, "v_genes"] = ImportHelper.strip_alleles(df, "v_genes")
        df.loc[:, "j_genes"] = ImportHelper.strip_alleles(df, "j_genes")

        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)

        return df

    @staticmethod
    def alternative_load_func(filename, params):
        df = airr.load_rearrangement(filename)
        df = ImportHelper.standardize_none_values(df)
        df.dropna(axis="columns", how="all", inplace=True)
        return df

    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["cell_id"]
        return ImportHelper.import_receptors(df, params)

    @staticmethod
    def get_documentation():
        doc = str(AIRRImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_types")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~source.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
