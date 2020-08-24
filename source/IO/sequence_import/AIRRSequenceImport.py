import airr
from pandas import DataFrame

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.import_parsers.ImportParser import ImportParser
from source.util.ImportHelper import ImportHelper


class AIRRSequenceImport:
    """
    Loads in the data to a list of sequences from AIRR format
    """
    @staticmethod
    def import_items(path, import_productive=True, import_with_stop_codon=False, import_out_of_frame=False,
                     region_type=RegionType.CDR3, region_definition=RegionDefinition.IMGT, column_mapping=None,
                     paired=False):
        if column_mapping is None:
            column_mapping = DefaultParamsLoader.load(ImportParser.keyword, "airr")["column_mapping"]

        params = {"import_productive": import_productive,
                  "import_with_stop_codon": import_with_stop_codon,
                  "import_out_of_frame": import_out_of_frame,
                  "region_type": region_type,
                  "region_definition": region_definition,
                  "column_mapping": column_mapping}

        if paired:
            raise NotImplementedError("AIRRSequenceImport: import of paired receptor data has not been implemented.")
        else:
            sequences = AIRRSequenceImport.import_all_sequences(path, params)

        return sequences

    @staticmethod
    def import_all_sequences(path, params: dict):
        df = airr.load_rearrangement(path)

        df.rename(columns=params["column_mapping"], inplace=True)

        df = ImportHelper.standardize_none_values(df)
        df = AIRRSequenceImport.preprocess_sequence_dataframe(df, params)
        sequences = df.apply(AIRRSequenceImport.import_sequence, axis=1).values
        return sequences

    @staticmethod
    def import_sequence(row):
        if "stop_codon" in row and row["stop_codon"] == False:
            frame_type = SequenceFrameType.STOP.name
        elif row["productive"]:
            frame_type = SequenceFrameType.IN.name
        elif "vj_in_frame" in row and row["vj_in_frame"] == True:
            frame_type = SequenceFrameType.IN.name
        else:
            frame_type = SequenceFrameType.OUT.name

        metadata = SequenceMetadata(v_gene=str(row["v_genes"]) if "v_genes" in row else None,
                                    j_gene=str(row["j_genes"]) if "j_genes" in row else None,
                                    chain=row["chains"] if "chains" in row else None,
                                    region_type=row["region_type"] if "region_type" in row else None,
                                    count=int(row["counts"]) if "counts" in row else None,
                                    frame_type=frame_type,
                                    custom_params={"rev_comp": row["rev_comp"]} if "rev_comp" in row else {})
        sequence = ReceptorSequence(amino_acid_sequence=str(row["sequence_aas"]) if "sequence_aas" in row else None,
                                    nucleotide_sequence=str(row["sequences"]) if "sequences" in row else None,
                                    identifier=str(row["sequence_identifiers"]) if "sequence_identifiers" in row else None,
                                    metadata=metadata)

        return sequence


    @staticmethod
    def preprocess_sequence_dataframe(df: DataFrame, params: dict):
        """
        Function for preprocessing data from a dataframe containing AIRR data, such that:
            - productive sequences, sequences with stop codons or out of frame sequences are filtered according to specification
            - if RegionType is CDR3, the leading C and trailing W are removed from the sequence to match the CDR3 definition
            - if no chain column was specified, the chain is extracted from the v gene name
            - the allele information is removed from the V and J genes
        """
        if params["import_with_stop_codon"] is False and "stop_codon" in df.columns:
            df = df[~df["stop_codon"]]
        if params["import_out_of_frame"] is False and "vj_in_frame" in df.columns:
            df = df[df["vj_in_frame"]]
        if params["import_productive"] is False and "productive" in df.columns:
            df = df[~df["productive"]]
        if params["import_with_stop_codon"] is False and params["import_out_of_frame"] is False:
            df = df[df["productive"]]

        ImportHelper.junction_to_cdr3(df, params["region_definition"], params["region_type"])

        if "chains" not in df.columns:
            df["chains"] = ImportHelper.load_chains_from_genes(df, "v_genes")

        df["v_genes"] = ImportHelper.strip_alleles(df, "v_genes")
        df["j_genes"] = ImportHelper.strip_alleles(df, "j_genes")

        df["region_type"] = params["region_type"].name

        return df

