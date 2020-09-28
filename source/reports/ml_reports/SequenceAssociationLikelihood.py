from typing import Tuple

from scipy.stats import beta

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.MLMethod import MLMethod
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder


class SequenceAssociationLikelihood(MLReport):
    """
    Plots the beta distribution used as a prior for class assignment in ProbabilisticBinaryClassifier. The distribution plotted shows
    the probability that a sequence is associated with a given class for a label.

    Attributes: the report does not take in any arguments.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sequence_assoc_report: SequenceAssociationLikelihood

    """

    DISTRIBUTION_PERCENTAGE_TO_SHOW = 0.999

    @classmethod
    def build_object(cls, **kwargs):
        return SequenceAssociationLikelihood(**kwargs)

    def check_prerequisites(self):
        if not isinstance(self.method, ProbabilisticBinaryClassifier):
            return False
        else:
            return True

    def __init__(self, method: MLMethod = None, result_path: str = None, name: str = None, **kwargs):
        super().__init__()
        self.method = method
        self.result_path = result_path
        self.name = name
        self.result_name = None

    def generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        upper_limit, lower_limit = self.get_distribution_limits()
        self.result_name = "beta_distribution"

        report_output_fig = self._safe_plot(upper_limit=upper_limit, lower_limit=lower_limit, output_written=False)
        output_figures = [] if report_output_fig is None else [report_output_fig]

        return ReportResult(name="Beta distribution priors - probability that a sequence is disease-associated",
                            output_figures=output_figures)

    def _plot(self, upper_limit, lower_limit):
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import STAP

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/StatDistributionPlot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_beta_distribution_binary_class(alpha0=self.method.alpha_0, beta0=self.method.beta_0,
                                                 alpha1=self.method.alpha_1, beta1=self.method.beta_1,
                                                 x_label=f"probability that receptor sequence is {self.method.label_name}-associated",
                                                 label0=f"{self.method.label_name} {self.method.class_mapping[0]}",
                                                 label1=f"{self.method.label_name} {self.method.class_mapping[1]}",
                                                 upper_limit=upper_limit, lower_limit=lower_limit,
                                                 result_path=self.result_path,
                                                 result_name=self.result_name)

        return ReportOutput(f"{self.result_path}{self.result_name}.pdf")

    def get_distribution_limits(self) -> Tuple[float, float]:
        lower_limit_0, upper_limit_0 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_0, self.method.beta_0)
        lower_limit_1, upper_limit_1 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_1, self.method.beta_1)
        lower_limit = min(lower_limit_0, lower_limit_1)
        upper_limit = max(upper_limit_0, upper_limit_1)

        return lower_limit, upper_limit
