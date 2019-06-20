from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.import_parsers.PreprocessingParameterParser import PreprocessingParameterParser


class DatasetChainFilterParser(PreprocessingParameterParser):

    @staticmethod
    def parse(params: dict):
        params = {**DefaultParamsLoader.load("preprocessing/", "DatasetChainFilter"), **params}

        return params, params
