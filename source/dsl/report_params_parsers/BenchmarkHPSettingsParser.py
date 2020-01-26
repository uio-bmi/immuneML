import warnings

from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.report_params_parsers.ErrorBarMeaning import ErrorBarMeaning

class BenchmarkHPSettingsParser:
    """
    The definition for BenchmarkHPSettings has the following format in the DSL:

    .. highlight:: yaml
    .. code-block:: yaml

        benchmark_report:
            type: BenchmarkHPSettingsParser
            params:
                errorbar_meaning: standard_error

    """
    ERRORBAR_DEFAULT = ErrorBarMeaning.STANDARD_ERROR

    @staticmethod
    def parse(params: dict):
        defaults = DefaultParamsLoader.load("reports/", "BenchmarkHPSettings")
        parsed = {**defaults, **params}

        try:
            parsed["errorbar_meaning"] = ErrorBarMeaning(parsed["errorbar_meaning"])
        except ValueError:
            warnings.warn("BenchmarkHPSettingsParser: unknown errorbar_meaning: '{}', will use '{}' instead.".format(params["errorbar_meaning"],
                                                                                                                     defaults["errorbar_meaning"]))
            parsed["errorbar_meaning"] = ErrorBarMeaning(defaults["errorbar_meaning"])

        return parsed, params
