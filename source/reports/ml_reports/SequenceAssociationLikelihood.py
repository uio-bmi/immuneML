from typing import Tuple

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP
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

        definitions:
            datasets:
                my_data:
                    ...
            encodings:
                enc1: SequenceAbundanceEncoder
            reports:
                report1: SequenceAssociationLikelihood
            ml_methods:
                ml1: ProbabilisticBinaryClassifier

        instructions:
            instruction_1:
                type: HPOptimization
                settings: [{encoding: enc1, ml_method: ml1]
                dataset: my_data
                assessment:
                    split_strategy: random
                    split_count: 1
                    training_percentage: 0.7
                    reports:
                        optimal_model:
                            - report1
                selection:
                    split_strategy: random
                    split_count: 1
                    training_percentage: 0.7
                    reports:
                        model:
                            - report1
                labels:
                  - CMV
                strategy: GridSearch
                metrics: [accuracy, auc]
                optimization_metric: accuracy
                batch_size: 4
                reports: []
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

    def generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        upper_limit, lower_limit = self.get_distribution_limits()
        result_name = "beta_distribution"

        pandas2ri.activate()

        with open(EnvironmentSettings.root_path + "source/visualization/StatDistributionPlot.R") as f:
            string = f.read()

        plot = STAP(string, "plot")

        plot.plot_beta_distribution_binary_class(alpha0=self.method.alpha_0, beta0=self.method.beta_0,
                                                 alpha1=self.method.alpha_1, beta1=self.method.beta_1,
                                                 x_label=f"probability that receptor sequence is {self.method.label_name}-associated",
                                                 label0=f"{self.method.label_name} {self.method.class_mapping[0]}",
                                                 label1=f"{self.method.label_name} {self.method.class_mapping[1]}",
                                                 upper_limit=upper_limit, lower_limit=lower_limit, result_path=self.result_path,
                                                 result_name=result_name)

        return ReportResult(name="Beta distribution priors - probability that a sequence is disease-associated",
                            output_figures=[ReportOutput(f"{self.result_path}{result_name}.pdf")])

    def get_distribution_limits(self) -> Tuple[float, float]:
        lower_limit_0, upper_limit_0 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_0, self.method.beta_0)
        lower_limit_1, upper_limit_1 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_1, self.method.beta_1)
        lower_limit = min(lower_limit_0, lower_limit_1)
        upper_limit = max(upper_limit_0, upper_limit_1)

        return lower_limit, upper_limit
