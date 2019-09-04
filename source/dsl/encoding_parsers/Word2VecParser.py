import copy

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.encodings.word2vec.model_creator.ModelType import ModelType


class Word2VecParser(EncodingParameterParser):

    @staticmethod
    def check_parameters(params: dict):
        assert "k" in params.keys() and isinstance(params["k"], int) and 0 < params["k"], \
            "Word2VecParser: k-mer length (parameter k) is not correctly specified."

        assert "vector_size" in params.keys() and isinstance(params["vector_size"], int) and 0 < params["k"], \
            "Word2VecParser: vector size is not correctly specified."

    @staticmethod
    def parse(params: dict):

        defaults = DefaultParamsLoader.load("encodings/", "Word2Vec")
        parsed = {**defaults, **params}
        Word2VecParser.check_parameters(parsed)

        specs = copy.deepcopy(parsed)
        parsed["model_type"] = ModelType[parsed["model_type"].upper()]

        return parsed, specs
