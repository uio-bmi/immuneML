import warnings
from pathlib import Path
from typing import List

import pickle
import pandas as pd
import plotly.express as px
import numpy as np

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.encodings.abundance_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
from immuneML.encodings.abundance_encoding.KmerAbundanceEncoder import KmerAbundanceEncoder
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SignificantFeaturesHelper import SignificantFeaturesHelper


class SignificantFeatures(DataReport):
    """
    Plots a boxplot of the number of significant features (label-associated k-mers or sequences) per Repertoire according to Fisher's exact test,
    across different classes for the given label.

    Internally uses the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder` for calculating
    significant k-mers, and
    :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
    :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
    to calculate significant full sequences (depending on whether the argument compairr_path was set).

    Specification arguments:

    - p_values (list): The p value thresholds to be used by Fisher's exact test. Each p-value specified here will become
      one panel in the output figure.

    - k_values (list): Length of the k-mers (number of amino acids) created by the
      :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`.
      When using a full sequence encoding (:py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
      :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`), specify 'full_sequence' here.
      Each value specified under k_values will represent one boxplot in the output figure.

    - label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.

    - compairr_path (str): If 'full_sequence' is listed under k_values, the path to the CompAIRR executable may be provided.
      If the compairr_path is specified, the :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
      will be used to compute the significant sequences. If the path is not specified and 'full_sequence' is listed under
      k-values, :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` will be used.

    - log_scale (bool): Whether to plot the y axis in log10 scale (log_scale = True) or continuous scale (log_scale = False). By default, log_scale is False.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_significant_features_report:
            SignificantFeatures:
                p_values:
                    - 0.1
                    - 0.01
                    - 0.001
                    - 0.0001
                k_values:
                    - 3
                    - 4
                    - 5
                    - full_sequence
                compairr_path: path/to/compairr # can be specified if 'full_sequence' is listed under k_values
                label: # Define a label, and the positive class for that given label
                    CMV:
                        positive_class: +
                log_scale: False

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = SignificantFeatures.__name__

        kwargs = SignificantFeaturesHelper.parse_parameters(kwargs, location)

        ParameterValidator.assert_keys_present(kwargs.keys(), ["log_scale"], location, location)
        ParameterValidator.assert_type_and_value(kwargs["log_scale"], bool, "SignificantFeatures", "log_scale")

        return SignificantFeatures(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, p_values: List[float] = None, k_values: List[int] = None,
                 label: dict = None, compairr_path: Path = None, log_scale: bool = False, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.p_values = p_values
        self.k_values = k_values
        self.label = label
        self.label_config = None
        self.compairr_path = compairr_path
        self.log_scale = log_scale

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{SignificantFeatures.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, SignificantFeatures.__name__, f"{SignificantFeatures.__name__}/label")

        plotting_data = self._compute_plotting_data()
        table_result = self._write_results_table(plotting_data)

        report_output_fig = self._safe_plot(plotting_data=plotting_data)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(name=self.name,
                            info="The number of significant features (label-associated k-mers or sequences) per Repertoire according to Fisher's exact test, across different classes for the given label.",
                            output_figures=output_figures,
                            output_tables=[table_result])

    def _compute_plotting_data(self):
        result = {"encoding": [],
                  "p-value": [],
                  "class": [],
                  "significant_features": []}

        positive_class, negative_class = self._get_positive_negative_classes()

        for k in self.k_values:
            encoder_name = SignificantFeaturesHelper._get_encoder_name(k)

            for p_value in self.p_values:
                pos_class_feature_counts, neg_class_feature_counts = self._compute_significant_feature_counts(k, p_value)
                n_examples = len(pos_class_feature_counts) + len(neg_class_feature_counts)

                result["encoding"].extend([encoder_name] * n_examples)
                result["p-value"].extend([p_value] * n_examples)
                result["class"].extend(
                    [positive_class] * len(pos_class_feature_counts) + [negative_class] * len(neg_class_feature_counts))
                result["significant_features"].extend(list(pos_class_feature_counts) + list(neg_class_feature_counts))

        return pd.DataFrame(result)

    def _get_positive_negative_classes(self):
        label = self.label_config.get_label_objects()[0]
        positive_class = label.positive_class
        negative_class = label.get_binary_negative_class()

        return positive_class, negative_class

    def _get_encoder_result_path(self, k, p_value):
        result_path = self.result_path / f"{SignificantFeaturesHelper._get_encoder_name(k)}_{p_value}"
        PathBuilder.build(result_path)
        return result_path

    def _write_results_table(self, data) -> ReportOutput:
        table_path = self.result_path / f"significant_features_report.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "Significant features across different Repertoire classes")

    def _plot(self, plotting_data):
        figure = px.box(plotting_data, x="encoding", y="significant_features", color="class",
                        facet_row=None, facet_col="p-value", log_y=self.log_scale,
                        labels={
                            "significant_features": "Number of significant features per AIRR according to Fisher's exact test",
                            "encoding": "Encoding",
                            "class": "Repertoire class"
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"significant_features_figure.html"

        figure.write_html(str(file_path))

        return ReportOutput(file_path, name="Significant features across different Repertoire classes")

    def _compute_significant_feature_counts(self, k, p_value):
        encoder_result_path = self._get_encoder_result_path(k, p_value)
        encoder_params = SignificantFeaturesHelper._build_encoder_params(self.label_config, encoder_result_path)

        if type(k) == int:
            pos_class_feature_counts, neg_class_feature_counts = self._compute_significant_kmer_counts(k, p_value, encoder_params)
        elif self.compairr_path is None:
            pos_class_feature_counts, neg_class_feature_counts = self._compute_significant_sequence_counts(p_value, encoder_params)
        else:
            pos_class_feature_counts, neg_class_feature_counts = self._compute_significant_compairr_sequence_counts(p_value, encoder_params)

        return pos_class_feature_counts, neg_class_feature_counts

    def _compute_significant_kmer_counts(self, k, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_kmer_encoder(self.dataset, k, p_value, encoder_params)

        with encoder.relevant_indices_path.open("rb") as file:
            relevant_indices = pickle.load(file)

        relevant_feature_presence = np.sum(encoder.kmer_presence_matrix[relevant_indices], axis=0)

        return self._get_positive_negative_class(relevant_feature_presence, encoder.matrix_repertoire_ids)

    def _compute_significant_sequence_counts(self, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_sequence_encoder(self.dataset, p_value, encoder_params)

        with encoder.relevant_indices_path.open("rb") as file:
            relevant_indices = pickle.load(file)

        relevant_feature_presence = self._sum_sequence_vectors_iterable(relevant_indices, encoder.comparison_data.get_item_vectors())

        return self._get_positive_negative_class(relevant_feature_presence, self.dataset.get_repertoire_ids())

    def _compute_significant_compairr_sequence_counts(self, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_compairr_sequence_encoder(self.dataset, p_value, encoder_params, self.compairr_path)

        with encoder.relevant_indices_path.open("rb") as file:
            relevant_indices = pickle.load(file)

        relevant_feature_presence = self._sum_sequence_vectors_iterable(relevant_indices, encoder.compairr_sequence_presence)

        return self._get_positive_negative_class(relevant_feature_presence, self.dataset.get_repertoire_ids())

    def _sum_sequence_vectors_iterable(self, relevant_indices, sequence_vector_iterable):
        relevant_feature_presence = np.zeros(shape=(6,))

        for i, sequence_vector in enumerate(sequence_vector_iterable):
            if relevant_indices[i]:
                relevant_feature_presence += sequence_vector

        return relevant_feature_presence

    def _get_relevant_feature_presence(self, encoder, relevant_indices):

        if isinstance(encoder, KmerAbundanceEncoder):
            relevant_feature_presence = np.sum(encoder.kmer_presence_matrix[relevant_indices], axis=0)
        elif isinstance(encoder, CompAIRRSequenceAbundanceEncoder):
            relevant_feature_presence = np.sum(encoder.sequence_presence_matrix[relevant_indices], axis=0)
        else:
            relevant_feature_presence = np.zeros(shape=(6,))

            for i, sequence_vector in enumerate(encoder.comparison_data.get_item_vectors()):
                if relevant_indices[i]:
                    relevant_feature_presence += sequence_vector

        return relevant_feature_presence

    def _get_positive_negative_class(self, relevant_feature_presence, repertoire_ids):
        is_positive_class = AbundanceEncoderHelper.check_is_positive_class(self.dataset, repertoire_ids, self.label_config)

        pos_class_feature_counts = relevant_feature_presence[is_positive_class]
        neg_class_feature_counts = relevant_feature_presence[np.logical_not(is_positive_class)]

        return pos_class_feature_counts, neg_class_feature_counts