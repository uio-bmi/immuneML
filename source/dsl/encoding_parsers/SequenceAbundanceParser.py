from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser


class SequenceAbundanceParser(EncodingParameterParser):
    """
    SequenceAbundanceEncoder requires the following params with possible values:
        comparison_attributes: [amino_acid_sequence]
        p_value_threshold: 0.05
        pool_size: 4

    The values written here are the default values, so it is also possible to have encoding without explicitly giving the parameters.
    """

    @staticmethod
    def parse(params: dict):
        defaults = DefaultParamsLoader.load("encodings/", "SequenceAbundance")
        parsed = {**defaults, **params}

        assert isinstance(parsed["comparison_attributes"], list) and all(isinstance(item, str) for item in parsed["comparison_attributes"]), \
            "SequenceAbundanceParser: all comparison attributes have to be strings."
        assert isinstance(parsed["p_value_threshold"], float), "SequenceAbundanceParser: p_value_threshold has to be of type float."

        return parsed, parsed
