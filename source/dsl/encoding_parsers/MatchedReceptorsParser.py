from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReceptorsParser(EncodingParameterParser):

    @staticmethod
    def parse(params: dict):
        defaults = DefaultParamsLoader.load("encodings/", "MatchedReceptors")
        parsed = {**defaults, **params}

        assert "reference_sequences" in params.keys(), "MatchedReceptorsParser: set reference sequences and try again."
        assert all([item in params["reference_sequences"].keys() for item in ["path", "format", "paired"]]), \
            "MatchedReceptorsParser: set 'path', 'format' and 'paired' for reference sequences and try again."
        assert isinstance(params["reference_sequences"]["format"], str) and \
               params["reference_sequences"]["format"].lower() in ["vdjdb", "iris"], \
            "MatchedReceptorsParser: reference sequences are accepted only in VDJdb and IRIS formats."
        assert params["reference_sequences"]["paired"] is True, "MatchedReceptorsParser: reference sequences must be paired."

        seqs = ReflectionHandler.get_class_by_name(
            "{}SequenceImport".format(params["reference_sequences"]["format"])) \
            .import_items(params["reference_sequences"]["path"], paired=True)

        parsed = {"reference_sequences": seqs,
                  "one_file_per_donor": parsed["one_file_per_donor"]}

        specs = {**parsed, **{"reference_sequences": params["reference_sequences"]}}

        return parsed, specs





