from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.report_params_parsers.CoefficientPlottingSetting import CoefficientPlottingSetting
from numbers import Number


class CoefficientsParser:
    """
    The definition for Coefficients has the following format in the DSL:

    .. highlight:: yaml
    .. code-block:: yaml

        lr_report:
            type: Coefficients
            params:
                coefs_to_plot: [all, nonzero, cutoff, n_largest] # Choose one or more
                cutoff: [0.2, 0.4]  # if 'cutoff' is in 'which_coefs', use this (absolute value) cutoff to select coefficients
                n_largest: [10]  # if 'n_largest' is in 'which_coefs', plot this number of largest coefficients

    """

    @staticmethod
    def parse(params: dict):
        defaults = DefaultParamsLoader.load("reports/", "Coefficients")
        parsed = {**defaults, **params}

        parsed["coefs_to_plot"] = CoefficientsParser._parse_plot_items(parsed)

        if CoefficientPlottingSetting.CUTOFF in parsed["coefs_to_plot"]:
            CoefficientsParser._check_numbers(parsed, "cutoff", "numeric")

        if CoefficientPlottingSetting.N_LARGEST in parsed["coefs_to_plot"]:
            CoefficientsParser._check_numbers(parsed, "n_largest", "integer")

        return parsed, params


    @staticmethod
    def _parse_plot_items(params):
        try:
            parsed_settings = [CoefficientPlottingSetting(keyword) for keyword in params["coefs_to_plot"]]
        except ValueError:
            raise ValueError("CoefficientsParser: an incorrect setting was specified in coefs_to_plot, "
                             "please refer to CoefficientPlottingSetting to see the possible options. ")

        return parsed_settings


    @staticmethod
    def _check_numbers(params, param_name, number_type="numeric"):
        instance_class = int if number_type == "integer" else Number

        assert len(params[param_name]) > 0, "CoefficientsParser: {} was specified in 'coefs_to_plot' but no values are specified for {}".format(param_name, param_name)

        for value in params[param_name]:
            assert isinstance(value, instance_class), "CoefficientsParser: {} must be a {} value.".format(param_name, number_type)
            assert value > 0, "CoefficientsParser: {} must be a positive value".format(param_name)


