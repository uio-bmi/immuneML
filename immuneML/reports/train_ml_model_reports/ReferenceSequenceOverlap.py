import logging
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2

from immuneML.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.encodings.filtered_sequence_encoding.SequenceCountEncoder import SequenceCountEncoder
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ReferenceSequenceOverlap(TrainMLModelReport):
    """
    The ReferenceSequenceOverlap report compares a list of disease-associated sequences produced by :ref:`SequenceAbundance` or :ref:`SequenceCount` encoders to
    a list of reference receptor sequences. It outputs a Venn diagram and a list of receptor sequences found both in the encoder and reference.

    The report compares the sequences by their sequence content and the additional comparison_attributes (such as V or J gene), as specified by the user.

    Arguments:

        reference_path (str): path to the reference file in csv format which contains one entry per row and has columns that correspond to the attributes
        listed under comparison_attributes argument

        comparison_attributes (list): list of attributes to use for comparison; all of them have to be present in the reference file where they should
        be the names of the columns

        label (str): name of the label for which the reference sequences should be compared to the model; if none, it takes the one label from the
        instruction; if it is none and multiple labels were specified for the instruction, the report will not be generated

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports: # the report is defined with all other reports under definitions/reports
            my_reference_overlap_report:
                ReferenceSequenceOverlap:
                    reference_path: reference.csv # a reference file with columns listed under comparison_attributes
                    comparison_attributes:
                        - sequence_aas
                        - v_genes
                        - j_genes

    """

    @classmethod
    def build_object(cls, **kwargs):

        ParameterValidator.assert_keys(kwargs.keys(), ['reference_path', 'comparison_attributes', 'name', 'label'], ReferenceSequenceOverlap.__name__,
                                       f"reports: {kwargs['name'] if 'name' in kwargs else ''}")

        kwargs['reference_path'] = Path(kwargs['reference_path'])

        assert kwargs['reference_path'].is_file(), f"{ReferenceSequenceOverlap.__name__}: 'reference_path' for report {kwargs['name']} is not " \
                                                         f"a valid file path."

        reference_sequences_df = pd.read_csv(kwargs['reference_path'])
        attributes = reference_sequences_df.columns.tolist()

        ParameterValidator.assert_keys_present(expected_values=kwargs['comparison_attributes'], values=attributes,
                                               location=ReferenceSequenceOverlap.__name__,
                                               parameter_name='columns in file under reference_path')

        return ReferenceSequenceOverlap(**kwargs)

    def __init__(self, reference_path: Path = None, comparison_attributes: list = None, name: str = None, state: TrainMLModelState = None,
                 result_path: Path = None, label: str = None):
        super().__init__(name)
        self.reference_path = reference_path
        self.comparison_attributes = comparison_attributes
        self.state = state
        self.result_path = result_path
        self.label = label

    def _generate(self) -> ReportResult:

        figures, tables = [], []

        PathBuilder.build(self.result_path)

        if ReferenceSequenceOverlap._check_encoder_class(self.state.optimal_hp_items[self.label].encoder):
            figure, data = self._compute_optimal_model_overlap()
            figures.append(figure)
            tables.append(data)

        for assessment_state in self.state.assessment_states:
            encoder = assessment_state.label_states[self.label].optimal_assessment_item.encoder
            if ReferenceSequenceOverlap._check_encoder_class(encoder):
                figure_filename = self.result_path / f"assessment_split_{assessment_state.split_index + 1}_model_vs_reference_overlap_{self.label}.pdf"
                df_filename = self.result_path / f"assessment_split_{assessment_state.split_index + 1}_overlap_sequences_{self.label}"
                figure, data = self._compute_model_overlap(figure_filename, df_filename, encoder,
                                                           f"overlap sequences between the model for assessment split "
                                                           f"{assessment_state.split_index + 1} and reference list")
                figures.append(figure)
                tables.append(data)

        return ReportResult(self.name, output_figures=figures, output_tables=tables)

    @staticmethod
    def _check_encoder_class(encoder):
        return any(isinstance(encoder, cls) for cls in [SequenceAbundanceEncoder, SequenceCountEncoder])

    def check_prerequisites(self):

        valid = True

        if self.label is None:
            if len(self.state.label_configuration.get_labels_by_name()) != 1:

                logging.warning(f"{ReferenceSequenceOverlap.__name__}: label parameter for report {self.name} is None and it could not be inferred "
                                f"from other information available in the report. It can be inferred automatically if there is only one label "
                                f"specified in the analysis, but got {self.state.label_configuration.get_labels_by_name()} instead. Skipping this "
                                f"report...")
                valid = False
            else:
                self.label = self.state.label_configuration.get_labels_by_name()[0]

        return valid

    def _compute_optimal_model_overlap(self) -> Tuple[ReportOutput, ReportOutput]:

        filename = self.result_path / f"optimal_model_vs_reference_overlap_{self.label}.pdf"
        df_filename = self.result_path / f"overlap_sequences_{self.label}.csv"
        encoder = self.state.optimal_hp_items[self.label].encoder

        return self._compute_model_overlap(filename, df_filename, encoder,
                                           f"overlap sequences between the reference and the optimal model for label {self.label}")

    def _compute_model_overlap(self, figure_filename, df_filename, encoder, name):

        reference_sequences_df = pd.read_csv(self.reference_path, usecols=self.comparison_attributes)
        reference_sequences = list(reference_sequences_df.to_records(index=False))
        attributes = reference_sequences_df.columns.tolist()

        model_sequences = self._extract_from_model(encoder)

        overlap_sequences = [sequence for sequence in model_sequences if sequence in reference_sequences]
        count_overlap = len(overlap_sequences)
        count_ref_only = len([sequence for sequence in reference_sequences if sequence not in model_sequences])
        count_model_only = len([sequence for sequence in model_sequences if sequence not in reference_sequences])

        self._make_venn_diagram(count_ref_only, count_overlap, count_model_only, 'reference', 'model', figure_filename)
        figure = ReportOutput(figure_filename, name)

        pd.DataFrame.from_records(overlap_sequences, columns=attributes).to_csv(df_filename, index=False)
        data = ReportOutput(df_filename, name)

        return figure, data

    def _extract_from_model(self, encoder):

        model_sequences_df = pd.read_csv(getattr(encoder, "relevant_sequence_csv_path"))
        model_attributes = model_sequences_df.columns.tolist()
        assert all(attribute in self.comparison_attributes for attribute in model_attributes), \
            f"{ReferenceSequenceOverlap.__name__}: comparison attributes from the report {self.name} ({self.comparison_attributes}) and from the optimal " \
            f"encoding {encoder.name} ({model_attributes}) do not match."

        return list(model_sequences_df[self.comparison_attributes].to_records(index=False))

    def _make_venn_diagram(self, count_ref_only: int, count_overlap: int, count_model_only: int, label_reference: str, label_model: str,
                           filename: str):
        subsets = Counter({"01": count_model_only, "10": count_ref_only, "11": count_overlap})
        diagram = venn2(subsets=subsets, set_labels=(label_reference, label_model), set_colors=('#72AAA1', '#E5B9AD'), alpha=0.8)
        for index in subsets:
            if subsets[index] == 0 and diagram.get_label_by_id(index) is not None:
                diagram.get_label_by_id(index).set_text("")
        plt.savefig(filename)
        plt.clf()
