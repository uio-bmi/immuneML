from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.import_parsers.PreprocessingParameterParser import PreprocessingParameterParser


class ClonotypeCountFilterParser(PreprocessingParameterParser):

    @staticmethod
    def parse(params: dict):

        params = {**DefaultParamsLoader.load("preprocessing/", "ClonotypeCountFilter"), **params}

        return params, params
