from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.util.ReflectionHandler import ReflectionHandler


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

        seqs = ReflectionHandler.get_class_by_name("{}SequenceImport".format(params["reference_sequences"]["format"]))\
            .import_items(params["reference_sequences"]["path"])

        parsed = {
            "reference_sequences": seqs,
            "max_distance": parsed["max_distance"],
            "summary": SequenceMatchingSummaryType[parsed["summary"].upper()]
        }

        specs = {**parsed, **{"reference_sequences": params["reference_sequences"]}}

        return parsed, specs
