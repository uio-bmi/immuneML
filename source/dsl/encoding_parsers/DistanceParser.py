import copy

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.encoding_parsers.EncodingParameterParser import EncodingParameterParser
from source.encodings.distance_encoding.DistanceMetricType import DistanceMetricType


class DistanceParser(EncodingParameterParser):

    @staticmethod
    def parse(params: dict):
        model_params = {**DefaultParamsLoader.load("encodings/", "Distance"), **params}
        specs = copy.deepcopy(model_params)

        model_params["distance_metric"] = DistanceMetricType[model_params["distance_metric"].upper()]

        return model_params, specs
