from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.encodings.word2vec.model_creator.ModelType import ModelType


class Word2VecParser(EncodingParameterParser):

    @staticmethod
    def check_parameters(params: dict):
        assert "k" in params.keys() and isinstance(params["k"], int) and 0 < params["k"], \
            "Word2VecParser: k-mer length (parameter k) is not correctly specified."

        assert "size" in params.keys() and isinstance(params["size"], int) and 0 < params["k"], \
            "Word2VecParser: vector size is not correctly specified."

    @staticmethod
    def parse(params: dict):
        Word2VecParser.check_parameters(params)

        parsed = {key: params[key] for key in params.keys()
                  if key not in ["model_creator"]}
        parsed["model_creator"] = ModelType[params["model_creator"].upper()]
        return parsed
