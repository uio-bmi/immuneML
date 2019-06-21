import copy

from source.analysis.criteria_matches.CriteriaTypeInstantiator import CriteriaTypeInstantiator
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.import_parsers.PreprocessingParameterParser import PreprocessingParameterParser


class MetadataFilterParser(PreprocessingParameterParser):

    @staticmethod
    def check_parameters(params: dict):
        assert "criteria" in params.keys(), \
            "Criteria for which repertoires to keep must be specified"

    @staticmethod
    def parse(params: dict):

        MetadataFilterParser.check_parameters(params)

        filter_params = {**DefaultParamsLoader.load("preprocessing/", "MetadataFilter"), **params}
        specs = copy.deepcopy(filter_params)
        filter_params["criteria"] = CriteriaTypeInstantiator.instantiate(filter_params["criteria"])

        return filter_params, specs
