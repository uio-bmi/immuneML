from pathlib import Path
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class SequenceAssociationLikelihood(MLReport):
    """
    Plots the beta distribution used as a prior for class assignment in ProbabilisticBinaryClassifier. The distribution plotted shows
    the probability that a sequence is associated with a given class for a label.

    Attributes: the report does not take in any arguments.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sequence_assoc_report: SequenceAssociationLikelihood

    """

    DISTRIBUTION_PERCENTAGE_TO_SHOW = 0.999
    STEP = 400

    @classmethod
    def build_object(cls, **kwargs):
        return SequenceAssociationLikelihood(**kwargs)

    def check_prerequisites(self):
        if not isinstance(self.method, ProbabilisticBinaryClassifier):
            return False
        else:
            return True

    def __init__(self, method: MLMethod = None, result_path: Path = None, name: str = None, **kwargs):
        super().__init__(method=method, result_path=result_path, name=name)
        self.method = method
        self.result_path = result_path
        self.name = name
        self.result_name = None

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        lower_limit, upper_limit = self.get_distribution_limits()
        self.result_name = "beta_distribution"

        report_output_fig = self._plot(upper_limit=upper_limit, lower_limit=lower_limit)
        output_figures = [] if report_output_fig is None else [report_output_fig]

        return ReportResult(name="Beta distribution priors - probability that a sequence is disease-associated",
                            output_figures=output_figures)

    def _plot(self, upper_limit, lower_limit):

        beta_distribution_x = np.arange(start=lower_limit, stop=upper_limit, step=(upper_limit - lower_limit) / SequenceAssociationLikelihood.STEP)
        negative_pdf = beta.pdf(beta_distribution_x, self.method.alpha_0, self.method.beta_0)
        positive_pdf = beta.pdf(beta_distribution_x, self.method.alpha_1, self.method.beta_1)

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=beta_distribution_x, y=negative_pdf, mode='lines', line=dict(color='#E69F00', width=2), name=f"{self.method.label_name} {self.method.class_mapping[0]}"))
        figure.add_trace(go.Scatter(x=beta_distribution_x, y=positive_pdf, mode='lines', line=dict(color='#0072B2', width=2), name=f"{self.method.label_name} {self.method.class_mapping[1]}"))

        figure.update_layout(template="plotly_white", xaxis_title=f"probability that receptor sequence is {self.method.label_name}-associated",
                             yaxis_title="probability density function", xaxis={'tickformat': '.2e'}, yaxis={'tickformat': '.2e'})

        output_path = self.result_path / f"{self.result_name}.html"

        figure.write_html(str(output_path))

        return ReportOutput(output_path)

    def get_distribution_limits(self) -> Tuple[float, float]:
        lower_limit_0, upper_limit_0 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_0, self.method.beta_0)
        lower_limit_1, upper_limit_1 = beta.interval(SequenceAssociationLikelihood.DISTRIBUTION_PERCENTAGE_TO_SHOW,
                                                     self.method.alpha_1, self.method.beta_1)
        lower_limit = min(lower_limit_0, lower_limit_1)
        upper_limit = max(upper_limit_0, upper_limit_1)

        return lower_limit, upper_limit
