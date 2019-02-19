import csv
import itertools
import pickle
from glob import iglob
from multiprocessing.pool import Pool

import pandas as pd
from pandas import DataFrame

from source.IO.DataLoader import DataLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.util.PathBuilder import PathBuilder


class MiXCRLoader(DataLoader):

    SAMPLE_ID = "sampleID"
    CLONE_COUNT = "cloneCount"
    PATIENT = "patient"
    V_GENES_WITH_SCORE = "allVHitsWithScore"
    J_GENES_WITH_SCORE = "allJHitsWithScore"
    CDR3_AA_SEQUENCE = "aaSeqCDR3"
    CDR3_NT_SEQUENCE = "nSeqCDR3"

    @staticmethod
    def load(path, params: dict = None) -> Dataset:
        PathBuilder.build(params["result_path"])
        filepaths = sorted(list(iglob(path + "**/*." + params["extension"], recursive=True)))
        dataset = MiXCRLoader._load(filepaths, params)
        return dataset

    @staticmethod
    def _load(filepaths: list, params: dict) -> Dataset:

        arguments = [(filepath, filepaths, params) for filepath in filepaths]

        with Pool(params["batch_size"]) as pool:
            repertoire_filenames = pool.starmap(MiXCRLoader._load_repertoire, arguments)

        dataset = Dataset(filenames=repertoire_filenames)
        return dataset

    @staticmethod
    def _load_repertoire(filepath, filepaths, params):

        index = filepaths.index(filepath)
        df = pd.read_csv(filepath, sep="\t")
        df.dropna(axis=1, how="all", inplace=True)

        sequences = MiXCRLoader._load_sequences(filepath, params, df)
        metadata = MiXCRLoader._extract_repertoire_metadata(filepath, params, df)
        patient_id = MiXCRLoader._extract_patient(filepath, df)
        repertoire = Repertoire(sequences=sequences, metadata=metadata, identifier=patient_id)
        filename = params["result_path"] + str(index) + ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(repertoire, file)

        return filename

    @staticmethod
    def _extract_patient(filepath: str, df):
        assert df[MiXCRLoader.PATIENT].nunique() == 1, "MiXCRLoader: multiple patients in a single file are not supported."
        patient_id = df[MiXCRLoader.PATIENT][0]
        return patient_id

    @staticmethod
    def _extract_repertoire_metadata(filepath, params, df):
        sample = MiXCRLoader._extract_sample_information(filepath, params, df)
        metadata = RepertoireMetadata(sample=sample)
        return metadata

    @staticmethod
    def _get_sample_id(df: DataFrame):
        assert df[MiXCRLoader.SAMPLE_ID].nunique() == 1, "MiXCRLoader: multiple sample IDs in a single file are not supported."
        return df[MiXCRLoader.SAMPLE_ID][0]

    @staticmethod
    def _extract_sample_information(filepath, params, df) -> Sample:
        sample = Sample(identifier=MiXCRLoader._get_sample_id(df))
        sample.custom_params = {}

        for param in params["custom_params"]:
            sample.custom_params[param["name"]] = MiXCRLoader._extract_custom_param(param, filepath, df)

        return sample

    @staticmethod
    def _extract_custom_param(param, filepath, df):
        if param["location"] == "filepath_binary":
            val = True if param["name"] in filepath and param["alternative"] not in filepath else False
        else:
            raise NotImplementedError
        return val

    @staticmethod
    def _load_sequences(filepath, params, df):
        sequences = []
        for index, row in df.iterrows():
            sequence = MiXCRLoader._process_row(filepath, df, row, params)
            sequences.append(sequence)
        return sequences

    @staticmethod
    def _process_row(filepath, df, row, params) -> ReceptorSequence:
        chain = MiXCRLoader._extract_chain(filepath)
        sequence_aa, sequence_nt = MiXCRLoader._extract_sequence_by_type(row, params)
        metadata = MiXCRLoader._extract_sequence_metadata(df, row, chain, params)
        sequence = ReceptorSequence(amino_acid_sequence=sequence_aa, nucleotide_sequence=sequence_nt, metadata=metadata)

        return sequence

    @staticmethod
    def _extract_chain(filepath: str):
        filename = filepath[filepath.rfind("/"):]
        return "A" if "TRA" in filename else "B" if "TRB" in filename else "NA"

    @staticmethod
    def _extract_v_gene(df, row):
        return row[MiXCRLoader.V_GENES_WITH_SCORE].split(",")[0]

    @staticmethod
    def _extract_j_gene(df, row):
        return row[MiXCRLoader.J_GENES_WITH_SCORE].split(",")[0]

    @staticmethod
    def _extract_sequence_metadata(df, row, chain, params):
        count = row[MiXCRLoader.CLONE_COUNT]
        v_gene = MiXCRLoader._extract_v_gene(df, row)
        j_gene = MiXCRLoader._extract_j_gene(df, row)
        region_type = params["sequence_type"]
        metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain, count=count, region_type=region_type)

        for column in df.keys():
            if column in params["additional_columns"]:
                metadata.custom_params[column] = row[column]

        return metadata

    @staticmethod
    def _extract_sequence_by_type(row, params):
        if params["sequence_type"] == "CDR3":
            sequence_aa = row[MiXCRLoader.CDR3_AA_SEQUENCE]
            sequence_nt = row[MiXCRLoader.CDR3_NT_SEQUENCE]
        else:
            raise NotImplementedError

        return sequence_aa, sequence_nt
