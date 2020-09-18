import pandas as pd

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.import_parsers.ImportParser import ImportParser
from source.util.ImportHelper import ImportHelper


class TenxGenomicsSequenceImport:
    """
    Loads in the data to a list of sequences from AIRR format
    """
    @staticmethod
    def import_items(path, import_productive=True, region_type=RegionType.CDR3, region_definition=RegionDefinition.IMGT,
                     column_mapping=None, paired=False):
        if column_mapping is None:
            column_mapping = DefaultParamsLoader.load(ImportParser.keyword, "airr")["column_mapping"]

        params = {"import_productive": import_productive,
                  "region_type": region_type,
                  "region_definition": region_definition,
                  "column_mapping": column_mapping}

        if paired:
            raise NotImplementedError("TenxGenomicsSequenceImport: import of paired receptor data has not yet been implemented.")
        else:
            sequences = TenxGenomicsSequenceImport.import_all_sequences(path, params)

        return sequences

    @staticmethod
    def import_all_sequences(path, params: dict):
        df = pd.read_csv(path, sep=",", iterator=False, dtype=str)

        df.rename(columns=params["column_mapping"], inplace=True)

        df = ImportHelper.standardize_none_values(df)

        if params["import_productive"]:
            df = df[df.productive == "True"]

        ImportHelper.junction_to_cdr3(df, params["region_definition"], params["region_type"])

        sequences = df.apply(TenxGenomicsSequenceImport.import_sequence, axis=1).values

        return sequences

    @staticmethod
    def import_sequence(row):
        if row["productive"]:
            frame_type = SequenceFrameType.IN.name
        else:
            frame_type = SequenceFrameType.OUT.name

        custom_params = {}
        for param in ["reads", "cell_id", "length", "full_length"]:
            if param in row:
                custom_params[param] = row[param]

        metadata = SequenceMetadata(v_gene=str(row["v_genes"]) if "v_genes" in row else None,
                                    j_gene=str(row["j_genes"]) if "j_genes" in row else None,
                                    chain=row["chains"] if "chains" in row else None,
                                    region_type=row["region_type"] if "region_type" in row else None,
                                    count=int(row["counts"]) if "counts" in row else None,
                                    frame_type=frame_type,
                                    custom_params=custom_params)
        sequence = ReceptorSequence(amino_acid_sequence=str(row["sequence_aas"]) if "sequence_aas" in row else None,
                                    nucleotide_sequence=str(row["sequences"]) if "sequences" in row else None,
                                    identifier=str(row["sequence_identifiers"]) if "sequence_identifiers" in row else None,
                                    metadata=metadata)
        return sequence


