from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType


class KmerFrequencyParser(EncodingParameterParser):

    @staticmethod
    def check_parameters(params: dict):
        assert "k" in params.keys() and isinstance(params["k"], int) and 0 < params["k"], \
            "KmerFrequencyParser: k-mer length (parameter k) is not correctly specified."

        assert all([isinstance(params[key], int) and params[key] >= 0 if key in params.keys() else True
                    for key in ["k_left", "k_right", "max_gap", "min_gap", "k"]]), \
            "KmerFrequencyParser: if given, parameters k_left, k_right, max_gap and min_gap have to be integers."

        assert all([params[key] > 0 if key in params.keys() else True for key in ["k_left", "k_right", "k"]]), \
            "KmerFrequencyParser: parameters k_left, k_right and k (k-mer lengths) have to be larger than 0 if given."

    @staticmethod
    def parse(params: dict):
        KmerFrequencyParser.check_parameters(params)

        model = {key: params[key] for key in params.keys()
                 if key not in ["normalization_type", "reads", "sequence_encoding_type"]}
        model["normalization_type"] = NormalizationType[params["normalization_type"].upper()]
        model["reads"] = ReadsType[params["reads"].upper()]
        model["sequence_encoding_type"] = SequenceEncodingType[params["sequence_encoding_type"].upper()]
        return model
