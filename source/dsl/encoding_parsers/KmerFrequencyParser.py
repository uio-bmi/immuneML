import copy

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType


class KmerFrequencyParser(EncodingParameterParser):

    @staticmethod
    def check_parameters(params: dict):
        assert all([isinstance(params[key], int) and params[key] >= 0 if key in params.keys() else True
                    for key in ["k_left", "k_right", "max_gap", "min_gap", "k"]]), \
            "KmerFrequencyParser: if given, parameters k_left, k_right, max_gap and min_gap have to be integers."

        assert all([params[key] > 0 if key in params.keys() else True for key in ["k_left", "k_right", "k"]]), \
            "KmerFrequencyParser: parameters k_left, k_right and k (k-mer lengths) have to be larger than 0 if given."

    @staticmethod
    def parse(params: dict):
        model_params = {**DefaultParamsLoader.load("encodings/", "KmerFrequency"), **params}
        specs = copy.deepcopy(model_params)

        KmerFrequencyParser.check_parameters(model_params)
        model_params["normalization_type"] = NormalizationType[model_params["normalization_type"].upper()]
        model_params["reads"] = ReadsType[model_params["reads"].upper()]
        model_params["sequence_encoding"] = SequenceEncodingType[model_params["sequence_encoding"].upper()]

        return model_params, specs
