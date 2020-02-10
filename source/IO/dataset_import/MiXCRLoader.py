import copy
import os
from glob import iglob, glob

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.IO.metadata_import.MetadataImport import MetadataImport
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.util.PathBuilder import PathBuilder


class MiXCRLoader(DataLoader):

    SAMPLE_ID = "sampleID"
    CLONE_COUNT = "cloneCount"
    PATIENT = "patient"
    V_GENES_WITH_SCORE = "allVHitsWithScore"
    V_HIT = "vHit"
    J_HIT = "jHit"
    J_GENES_WITH_SCORE = "allJHitsWithScore"
    CDR3_AA_SEQUENCE = "aaSeqCDR3"
    CDR3_NT_SEQUENCE = "nSeqCDR3"
    SEQUENCE_NAME_MAP = {
        "CDR3": {"AA": "aaSeqCDR3", "NT": "nSeqCDR3"},
        "CDR1": {"AA": "aaSeqCDR1", "NT": "nSeqCDR1"},
        "CDR2": {"AA": "aaSeqCDR2", "NT": "nSeqCDR2"},
        "FR1":  {"AA": "aaSeqFR1",  "NT": "nSeqFR1"},
        "FR2":  {"AA": "aaSeqFR2",  "NT": "nSeqFR2"},
        "FR3":  {"AA": "aaSeqFR3",  "NT": "nSeqFR3"},
        "FR4":  {"AA": "aaSeqFR4",  "NT": "nSeqFR4"}
    }

    def load(self, path, params: dict = None) -> RepertoireDataset:
        params = copy.deepcopy(params)
        PathBuilder.build(params["result_path"])
        filepaths = sorted(list(iglob(path + "**/*." + params["extension"], recursive=True)))

        if os.path.isfile(params["result_path"] + "dataset.pkl") and len(glob(params["result_path"])) == len(filepaths):
            dataset = PickleLoader.load(params["result_path"] + "dataset.pkl")
        else:
            if "metadata_file" in params:
                metadata = MetadataImport.import_metadata(params["metadata_file"])
                params["metadata"] = metadata
            dataset = MiXCRLoader._load(filepaths, params)
            PickleExporter.export(dataset, params["result_path"], "dataset.pkl")
        return dataset

    @staticmethod
    def _load(filepaths, params) -> RepertoireDataset:
        metadata_df = pd.read_csv(params["metadata_file"], sep=',')
        repertoires = []

        for index in range(len(filepaths)):
            repertoires.append(MiXCRLoader._load_repertoire(filepaths[index], params, metadata_df.iloc[index]))

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=params["metadata_file"], params={key: metadata_df[key].unique().tolist() for key in metadata_df.columns})
        return dataset

    @staticmethod
    def _load_repertoire(filepath: str, params: dict, metadata) -> SequenceRepertoire:
        df = pd.read_csv(filepath, delimiter='\t')
        df.dropna(axis=1, how="all", inplace=True)

        sequences_aas = df[MiXCRLoader.SEQUENCE_NAME_MAP[params["sequence_type"]]["AA"]]
        sequences = df[MiXCRLoader.SEQUENCE_NAME_MAP[params["sequence_type"]]["NT"]]
        if params["CDR3_type"] == "IMGT":
            sequences_aas = sequences_aas.str[1:-1]
            sequences = sequences.str[3:-3]

        repertoire = SequenceRepertoire.build(sequence_aas=sequences_aas.tolist(),
                                              sequences=sequences.tolist(),
                                              v_genes=MiXCRLoader._load_genes(df, MiXCRLoader.V_GENES_WITH_SCORE).tolist(),
                                              j_genes=MiXCRLoader._load_genes(df, MiXCRLoader.J_GENES_WITH_SCORE).tolist(),
                                              chains=MiXCRLoader._load_chains(df, MiXCRLoader.V_GENES_WITH_SCORE).tolist(),
                                              counts=df[MiXCRLoader.CLONE_COUNT].tolist(),
                                              region_types=[params["sequence_type"] for i in range(df.shape[0])],
                                              path=params["result_path"], metadata=metadata.to_dict(),
                                              custom_lists={}, sequence_identifiers=list(range(df.shape[0])))

        return repertoire

    @staticmethod
    def _load_chains(df: pd.DataFrame, column_name):
        tmp_df = df.apply(lambda row: Chain[[x for x in [chain.value for chain in Chain] if x in row[column_name]][0]]
                        if len([x for x in [chain.value for chain in Chain] if x in row[column_name]]) > 0 else None, axis=1)
        return tmp_df

    @staticmethod
    def _load_genes(df: pd.DataFrame, column_name):
        tmp_df = df.apply(lambda row: row[column_name].split(",")[0].replace("TRB", "").replace("TRA", "").split("*", 1)[0], axis=1)
        return tmp_df
