import functools
import operator

import pandas as pd

from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser


class MatchedReferenceParser(EncodingParameterParser):

    @staticmethod
    def parse(params: dict):

        defaults = DefaultParamsLoader.load("encodings/", "MatchedReference")
        parsed = {**defaults, **params}

        assert "reference_sequences" in params.keys(), "MatchedReferenceParser: set reference sequences and try again."
        assert all([item in params["reference_sequences"].keys() for item in ["path", "format"]]), \
            "MatchedReferenceParser: set path and format for reference sequences and try again."
        assert isinstance(params["reference_sequences"]["format"], str) and \
               params["reference_sequences"]["format"].lower() in ["vdjdb", "iris"], \
            "MatchedReferenceParser: reference sequences are accepted only in VDJdb and IRIS formats."

        seqs = getattr(MatchedReferenceParser,
                       "parse_{}".format(params["reference_sequences"]["format"].lower()))(params["reference_sequences"]["path"])

        parsed = {
            "reference_sequences": seqs,
            "max_distance": parsed["max_distance"],
            "summary": SequenceMatchingSummaryType[parsed["summary"].upper()]
        }

        specs = {**parsed, **{"reference_sequences": params["reference_sequences"]}}

        return parsed, specs

    @staticmethod
    def parse_iris(path: str) -> list:
        df = pd.read_csv(path, sep=";")
        df = df.where((pd.notnull(df)), None)

        sequences = df.apply(MatchedReferenceParser.process_iris_row, axis=1).values
        sequences = functools.reduce(operator.iconcat, sequences, [])

        return sequences

    @staticmethod
    def process_iris_row(row):
        sequences = []

        if row["Chain: TRA (1)"] is not None:
            sequences.extend(MatchedReferenceParser.process_iris_chain(row, "A"))
        if row["Chain: TRB (1)"] is not None:
            sequences.extend(MatchedReferenceParser.process_iris_chain(row, "B"))

        return sequences

    @staticmethod
    def process_iris_chain(row, chain):
        sequences = []

        v_genes = set([gene.split("*")[0].replace("TR{}".format(chain), "") for gene in
                       row["TR{} - V gene (1)".format(chain)].split(" | ")])
        j_genes = set([gene.split("*")[0].replace("TR{}".format(chain), "") for gene in
                       row["TR{} - J gene (1)".format(chain)].split(" | ")])

        for v_gene in v_genes:
            for j_gene in j_genes:
                metadata = SequenceMetadata(v_gene=v_gene, j_gene=j_gene, chain=chain)
                sequences.append(ReceptorSequence(amino_acid_sequence=row["Chain: TR{} (1)".format(chain)],
                                                  metadata=metadata))

        return sequences

    @staticmethod
    def parse_vdjdb(path: str) -> list:
        return VDJdbSequenceImport.import_sequences(path)