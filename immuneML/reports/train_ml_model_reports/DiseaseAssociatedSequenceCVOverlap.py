import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.express as px

from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
from immuneML.encodings.abundance_encoding.KmerAbundanceEncoder import KmerAbundanceEncoder
from immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SequenceAnalysisHelper import SequenceAnalysisHelper


class DiseaseAssociatedSequenceCVOverlap(TrainMLModelReport):
    """
    DiseaseAssociatedSequenceCVOverlap report makes one heatmap per label showing the overlap of disease-associated sequences (or k-mers)
    produced by the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
    :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder` or
    :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`
    between folds of cross-validation (either inner or outer loop of the nested CV). The overlap is computed by the following equation:

    .. math::

        overlap(X,Y) = \\frac{|X \\cap Y|}{min(|X|, |Y|)} x 100

    For details, see Greiff V, Menzel U, Miho E, et al. Systems Analysis Reveals High Genetic and Antigen-Driven Predetermination of Antibody
    Repertoires throughout B Cell Development. Cell Reports. 2017;19(7):1467-1478. doi:10.1016/j.celrep.2017.04.054.


    **Specification arguments:**

    - compare_in_selection (bool): whether to compute the overlap over the inner loop of the nested CV - the sequence overlap is shown across CV
      folds for the model chosen as optimal within that selection

    - compare_in_assessment (bool): whether to compute the overlap over the optimal models in the outer loop of the nested CV


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_overlap_report: DiseaseAssociatedSequenceCVOverlap # report has no parameters

    """
    COMPATIBLE_ENCODERS = (SequenceAbundanceEncoder, CompAIRRSequenceAbundanceEncoder, KmerAbundanceEncoder)

    @classmethod
    def build_object(cls, **kwargs):
        return DiseaseAssociatedSequenceCVOverlap(**kwargs)

    def __init__(self, state: TrainMLModelState = None, result_path: Path = None, name: str = None, compare_in_selection: bool = False,
                 compare_in_assessment: bool = False, label: Label = None, number_of_processes: int = 1):
        super().__init__(name=name, state=state, label=label, result_path=result_path, number_of_processes=number_of_processes)
        self.compare_in_selection = compare_in_selection
        self.compare_in_assessment = compare_in_assessment

    def _generate(self) -> ReportResult:

        assert self.state, (f"{self.__class__.__name__}: this report can only be used in combination with TrainMLModel "
                            f"instruction. The instruction output is not available here so the report will be skipped.")

        PathBuilder.build(self.result_path)

        tables, figures = [], []
        for label_name in self.state.label_configuration.get_labels_by_name():
            if self.compare_in_assessment:
                table, figure = self._generate_for_assessment(label_name)
                tables.append(table)
                figures.append(figure)
            if self.compare_in_selection:
                tmp_tables, tmp_figures = self._generate_for_selection(label_name)
                tables += tmp_tables
                figures += tmp_figures

        return ReportResult(self.name,
                            info="One heatmap per label showing the overlap of disease-associated sequences produced by the SequenceAbundance encoder between folds of cross-validation (either inner or outer loop of the nested CV)",
                            output_figures=[fig for fig in figures if fig is not None],
                            output_tables=[tab for tab in tables if tab is not None])

    def _generate_for_assessment(self, label_name: str):
        hp_items = [st.label_states[label_name].optimal_assessment_item for st in self.state.assessment_states
                    if st.label_states[label_name].optimal_assessment_item.encoder.__class__ in DiseaseAssociatedSequenceCVOverlap.COMPATIBLE_ENCODERS]
        table, figure = self._compute_overlap(hp_items, f'sequence_overlap_{label_name}_assessment')
        return table, figure

    def _generate_for_selection(self, label_name: str):
        tables, figures = [], []
        for assessment_index, assessment_state in enumerate(self.state.assessment_states):
            selection_state = assessment_state.label_states[label_name].selection_state
            if selection_state.optimal_hp_setting.encoder.__class__ in DiseaseAssociatedSequenceCVOverlap.COMPATIBLE_ENCODERS:
                hp_items = selection_state.hp_items[selection_state.optimal_hp_setting.get_key()]
                table, figure = self._compute_overlap(hp_items, f'sequence_overlap_{label_name}_selection_{assessment_index + 1}_split')
                tables.append(table)
                figures.append(figure)
        return tables, figures

    def _compute_overlap(self, hp_items: List[HPItem], filename: str) -> Tuple[ReportOutput, ReportOutput]:
        overlap_matrix = SequenceAnalysisHelper.compute_overlap_matrix(hp_items)

        if overlap_matrix is None:
            logging.warning(f'{DiseaseAssociatedSequenceCVOverlap.__name__}: overlap matrix is None, some of the relevant sequence sets were empty, '
                            f'no report will be made.')
            return None, None

        row_col_names = [f"{item.hp_setting}_split_{item.split_index+1}" for item in hp_items]
        table_output = self._export_matrix(overlap_matrix, filename, row_col_names)
        figure_output = self._make_figure(overlap_matrix, filename, row_col_names)
        return table_output, figure_output

    def _export_matrix(self, overlap_matrix, filename, row_col_names) -> ReportOutput:
        data_path = self.result_path / f"{filename}.csv"
        pd.DataFrame(overlap_matrix, columns=row_col_names, index=row_col_names).to_csv(data_path)
        return ReportOutput(data_path, " ".join(filename.split('_') + ['data']))

    def _make_figure(self, overlap_matrix, filename, row_col_names) -> ReportOutput:
        figure = px.imshow(overlap_matrix, x=row_col_names, y=row_col_names, zmin=0, zmax=100, color_continuous_scale=px.colors.sequential.Teal,
                           template='plotly_white')
        figure.update_traces(hovertemplate="Overlap of disease-associated<br>sequences between<br>%{x} and %{y}:<br>%{z}%<extra></extra>")
        figure_path = self.result_path / f"{filename}.html"
        figure.write_html(str(figure_path))
        return ReportOutput(figure_path, " ".join(filename.split('_')))
