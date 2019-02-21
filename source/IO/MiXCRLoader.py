import csv
import itertools
import pickle
import warnings
from glob import iglob
from multiprocessing.pool import Pool

import pandas as pd
from pandas import DataFrame

from source.IO.DataLoader import DataLoader
from source.IO.PickleExporter import PickleExporter
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
    V_HIT = "vHit"
    J_HIT = "jHit"
    J_GENES_WITH_SCORE = "allJHitsWithScore"
    CDR3_AA_SEQUENCE = "aaSeqCDR3"
    CDR3_NT_SEQUENCE = "nSeqCDR3"

    @staticmethod
    def load(path, params: dict = None) -> Dataset:
        PathBuilder.build(params["result_path"])
        filepaths = sorted(list(iglob(path + "**/*." + params["extension"], recursive=True)))
        dataset = MiXCRLoader._load(filepaths, params)
        PickleExporter.export(dataset, params["result_path"], "dataset.pkl")
        return dataset

    @staticmethod
    def _load(filepaths: list, params: dict) -> Dataset:

        arguments = [(filepath, filepaths, params) for filepath in filepaths]

        with Pool(params["batch_size"]) as pool:
            output = pool.starmap(MiXCRLoader._load_repertoire, arguments)

        repertoire_filenames = [out[0] for out in output]
        custom_params = MiXCRLoader._prepare_sample_custom_params([out[1] for out in output])

        dataset = Dataset(filenames=repertoire_filenames, params=custom_params)
        return dataset

    @staticmethod
    def _prepare_sample_custom_params(params) -> dict:
        custom_params = {}
        for p in params:
            for key in p.keys():
                if key in custom_params:
                    custom_params[key].add(p[key])
                else:
                    custom_params[key] = {p[key]}

        return custom_params

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

        return filename, metadata.sample.custom_params

    @staticmethod
    def _extract_patient(filepath: str, df):
        assert df[MiXCRLoader.PATIENT].nunique() == 1, \
            "MiXCRLoader: multiple patients in a single file are not supported. Issue with " + filepath
        return df[MiXCRLoader.PATIENT][0]

    @staticmethod
    def _extract_repertoire_metadata(filepath, params, df):
        sample = MiXCRLoader._extract_sample_information(filepath, params, df)
        metadata = RepertoireMetadata(sample=sample)
        return metadata

    @staticmethod
    def _get_sample_id(df: DataFrame, filepath):
        if df[MiXCRLoader.SAMPLE_ID].nunique() != 1:
            warnings.warn("MiXCRLoader: multiple sample IDs in a single file are not supported. Ignoring sample id... Issue with " + filepath, Warning)
            sample_id = None
        else:
            sample_id = df[MiXCRLoader.SAMPLE_ID][0]
        return sample_id

    @staticmethod
    def _extract_sample_information(filepath, params, df) -> Sample:
        sample = Sample(identifier=MiXCRLoader._get_sample_id(df, filepath))
        sample.custom_params = {}

        for param in params["custom_params"]:
            sample.custom_params[param["name"]] = MiXCRLoader._extract_custom_param(param, filepath)

        return sample

    @staticmethod
    def _extract_custom_param(param, filepath):
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
    def _extract_gene(row, fields):
        """
        :param df: dataframe
        :param row: dataframe row
        :param fields: list of field names sorted by preference
        :return: the field value in the row for the first matched field name from fields
        """
        i = 0
        gene = None
        while i < len(fields) and gene is None:
            if fields[i] in row and isinstance(row[fields[i]], str):
                gene = row[fields[i]].split(",")[0].replace("TRB", "").replace("TRA", "").split("*", 1)[0]
            i += 1

        return gene

    @staticmethod
    def _extract_sequence_metadata(df, row, chain, params):
        count = row[MiXCRLoader.CLONE_COUNT]
        v_gene = MiXCRLoader._extract_gene(row, [MiXCRLoader.V_HIT, MiXCRLoader.V_GENES_WITH_SCORE])
        j_gene = MiXCRLoader._extract_gene(row, [MiXCRLoader.J_HIT, MiXCRLoader.J_GENES_WITH_SCORE])
        region_type = params["sequence_type"]
        metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain, count=count, region_type=region_type)

        for column in df.keys():
            if params["additional_columns"] == "*" or column in params["additional_columns"]:
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
