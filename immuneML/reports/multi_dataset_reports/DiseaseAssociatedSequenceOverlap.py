from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px

from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.multi_dataset_reports.MultiDatasetReport import MultiDatasetReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SequenceAnalysisHelper import SequenceAnalysisHelper


class DiseaseAssociatedSequenceOverlap(MultiDatasetReport):
    """
    DiseaseAssociatedSequenceOverlap report makes a heatmap showing the overlap of disease-associated sequences produced by :ref:`SequenceAbundance` or :ref:`SequenceCount` encoders
    between multiple datasets of different sizes (different number of repertoires per dataset).

    This plot can be used only with MultiDatasetBenchmarkTool

    The overlap is computed by the following equation:

    .. math::

        overlap(X,Y) = \frac{|X \cap Y|}{min(|X|, |Y|)} x 100

    For details, see Greiff V, Menzel U, Miho E, et al. Systems Analysis Reveals High Genetic and Antigen-Driven Predetermination of Antibody
    Repertoires throughout B Cell Development. Cell Reports. 2017;19(7):1467-1478. doi:10.1016/j.celrep.2017.04.054.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports: # the report is defined with all other reports under definitions/reports
            my_overlap_report: DiseaseAssociatedSequenceOverlap # report has no parameters

    """

    @classmethod
    def build_object(cls, **kwargs):
        return DiseaseAssociatedSequenceOverlap(**kwargs)

    def __init__(self, instruction_states: List[TrainMLModelState] = None, name: str = None, result_path: Path = None):
        super().__init__(name)
        self.instruction_states = instruction_states
        self.result_path = result_path
        self.label = None

    def _generate(self) -> ReportResult:
        self.result_path = PathBuilder.build(self.result_path / self.name)
        self._extract_label()

        hp_items = [state.optimal_hp_items[self.label] for state in self.instruction_states]
        overlap_matrix = SequenceAnalysisHelper.compute_overlap_matrix(hp_items)

        labels = [state.dataset.name for state in self.instruction_states]
        figure_path = self._make_figure(overlap_matrix, labels)
        data_path = self._export_matrix(overlap_matrix, labels)

        return ReportResult(output_figures=[ReportOutput(figure_path, 'sequence overlap across datasets')],
                            output_tables=[ReportOutput(data_path, 'sequence overlap across datasets (csv)')])

    def _extract_label(self):
        all_labels = []
        for state in self.instruction_states:
            all_labels += state.label_configuration.get_labels_by_name()

        all_labels = set(all_labels)
        assert len(all_labels) == 1, \
            f"{DiseaseAssociatedSequenceOverlap.__name__}: multiple labels were specified {all_labels}, but this report accepts only one label."

        self.label = list(all_labels)[0]

    def _export_matrix(self, overlap_matrix, labels):
        data_path = self.result_path / "sequence_overlap.csv"
        pd.DataFrame(overlap_matrix, columns=labels, index=labels).to_csv(data_path)
        return data_path

    def _make_figure(self, overlap_matrix, labels):
        figure = px.imshow(overlap_matrix, x=labels, y=labels, zmin=0, zmax=100, color_continuous_scale=px.colors.sequential.Teal,
                           template='plotly_white')
        figure.update_traces(hovertemplate="Overlap of disease-associated<br>sequences between datasets<br>%{x} and %{y}:<br>%{z}%<extra></extra>")
        figure_path = self.result_path / "sequence_overlap.html"
        figure.write_html(str(figure_path))
        return figure_path

