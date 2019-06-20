import os
import pickle
from glob import iglob, glob
from multiprocessing.pool import Pool
import copy

import pandas as pd
from pandas import DataFrame

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.IO.metadata_import.MetadataImport import MetadataImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.Constants import Constants
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

    def load(self, path, params: dict = None) -> Dataset:
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
    def _load(filepaths: list, params: dict) -> Dataset:

        arguments = [(filepath, filepaths, params) for filepath in filepaths]

        with Pool(params["batch_size"]) as pool:
            output = pool.starmap(MiXCRLoader._load_repertoire, arguments)

        repertoire_filenames = [out[0] for out in output]
        custom_params = MiXCRLoader._prepare_custom_params([out[1] for out in output])

        dataset = Dataset(filenames=repertoire_filenames, params=custom_params, metadata_file=params["metadata_file"])
        return dataset

    @staticmethod
    def _prepare_custom_params(params: list) -> dict:
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

        metadata = MiXCRLoader._extract_repertoire_metadata(filepath, params)

        sequences = MiXCRLoader._load_sequences(filepath, params, df)
        patient_id = metadata.custom_params["donor"]
        repertoire = Repertoire(sequences=sequences, metadata=metadata, identifier=patient_id)
        filename = params["result_path"] + str(index) + ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(repertoire, file)

        return filename, metadata.custom_params

    @staticmethod
    def _extract_repertoire_metadata(filepath, params) -> RepertoireMetadata:
        metadata = [m for m in params["metadata"] if os.path.basename(m["rep_file"]) == os.path.basename(filepath)]
        metadata = metadata[0]["metadata"]
        return metadata

    @staticmethod
    def _get_sample_id(df: DataFrame):

        sample_id = None

        if MiXCRLoader.SAMPLE_ID in df.keys():

            if df[MiXCRLoader.SAMPLE_ID].nunique() != 1:
                sample_id = None
            else:
                sample_id = df[MiXCRLoader.SAMPLE_ID][0]

        return sample_id

    @staticmethod
    def _extract_sample_information(df) -> Sample:
        identifier = MiXCRLoader._get_sample_id(df)
        sample = None

        if identifier is not None:
            sample = Sample(identifier=identifier)

        return sample

    @staticmethod
    def _load_sequences(filepath, params, df):
        sequences = df.apply(MiXCRLoader._process_row, axis=1, args=(filepath, df, params, )).values
        return sequences

    @staticmethod
    def _process_row(row, filepath, df, params,) -> ReceptorSequence:
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
        :param row: row of the dataframe <=> one MiXCR line
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
        v_subgroup = v_gene.split("-")[0]
        j_subgroup = j_gene.split("-")[0]
        v_allele = v_gene + Constants.ALLELE_DELIMITER + Constants.UNKNOWN
        j_allele = j_gene + Constants.ALLELE_DELIMITER + Constants.UNKNOWN
        region_type = params["sequence_type"]
        metadata = SequenceMetadata(v_subgroup=v_subgroup, v_gene=v_gene, v_allele=v_allele, j_subgroup=j_subgroup, j_gene=j_gene, j_allele=j_allele, chain=chain, count=count, region_type=region_type)

        if MiXCRLoader.SAMPLE_ID in params["additional_columns"] and MiXCRLoader.SAMPLE_ID in df.keys():
            sample = Sample(identifier=row[MiXCRLoader.SAMPLE_ID])
            metadata.sample = sample

        for column in df.keys():
            if params["additional_columns"] == "*" or column in params["additional_columns"]:
                metadata.custom_params[column] = row[column]

        return metadata

    @staticmethod
    def _extract_sequence_by_type(row, params):

        column_names = params["sequence_type"].split("+")

        sequence_aa = "".join([row[MiXCRLoader.SEQUENCE_NAME_MAP[item]["AA"]][1:-1] if item == "CDR3" else row[MiXCRLoader.SEQUENCE_NAME_MAP[item]["AA"]] for item in column_names])
        sequence_nt = "".join([row[MiXCRLoader.SEQUENCE_NAME_MAP[item]["NT"]][3:-3] if item == "CDR3" else row[MiXCRLoader.SEQUENCE_NAME_MAP[item]["NT"]] for item in column_names])

        return sequence_aa, sequence_nt
