import warnings
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.KmerHelper import KmerHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SignificantFeaturesHelper import SignificantFeaturesHelper


class RecoveredSignificantFeatures(DataReport):
    """
    Compares a given collection of groundtruth implanted signals (sequences or k-mers) to the significant label-associated
    k-mers or sequences according to Fisher's exact test.

    Internally uses the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder` for calculating
    significant k-mers, and
    :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
    :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
    to calculate significant full sequences (depending on whether the argument compairr_path was set).

    This report creates two plots:

    - the first plot is a bar chart showing what percentage of the ground truth implanted signals were found to be significant.

    - the second plot is a bar chart showing what percentage of the k-mers/sequences found to be significant match the
      ground truth implanted signals.

    To compare k-mers or sequences of differing lengths, the ground truth sequences or long k-mers are split into k-mers
    of the given size through a sliding window approach. When comparing 'full_sequences' to ground truth sequences, a match
    is only registered if both sequences are of equal length.


    Specification arguments:

    - groundtruth_sequences_path (str): Path to a file containing the true implanted (sub)sequences, e.g., full sequences or k-mers.
      The file should contain one sequence per line, without a header, and without V or J genes.

    - trim_leading_trailing (bool): Whether to trim the leading and trailing first positions from the provided groundtruth sequences,
      e.g., the leading C and trailing Y/F amino acids.
      This is necessary for comparing full sequences when the main dataset is imported using settings that also trim
      the leading and trailing positions (specified by the region_type parameter). By default, trim_leading_trailing is False.

    - p_values (list): The p value thresholds to be used by Fisher's exact test. Each p-value specified here will become one panel in the output figure.

    - k_values (list): Length of the k-mers (number of amino acids) created by the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`.
      When using a full sequence encoding (:py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
      :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`), specify 'full_sequence' here.
      Each value specified under k_values will represent one bar in the output figure.

    - label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.

    - compairr_path (str): If 'full_sequence' is listed under k_values, the path to the CompAIRR executable may be provided.
      If the compairr_path is specified, the :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
      will be used to compute the significant sequences. If the path is not specified and 'full_sequence' is listed under
      k-values, :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` will be used.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_recovered_significant_features_report:
            RecoveredSignificantFeatures:
                groundtruth_sequences_path: path/to/groundtruth/sequences.txt
                trim_leading_trailing: False
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
    """

    @classmethod
    def build_object(cls, **kwargs):
        location = RecoveredSignificantFeatures.__name__

        kwargs = SignificantFeaturesHelper.parse_parameters(kwargs, location)
        kwargs = SignificantFeaturesHelper.parse_sequences_path(kwargs, "groundtruth_sequences_path", location)

        ParameterValidator.assert_keys_present(kwargs.keys(), ["trim_leading_trailing"], location, location)
        ParameterValidator.assert_type_and_value(kwargs["trim_leading_trailing"], bool, "RecoveredSignificantFeatures", "trim_leading_trailing")

        return RecoveredSignificantFeatures(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, groundtruth_sequences_path: Path = None,
                 trim_leading_trailing: bool = None,
                 p_values: List[float] = None, k_values: List[int] = None, label: dict = None,
                 compairr_path: Path = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.groundtruth_sequences_path = groundtruth_sequences_path
        self.trim_leading_trailing = trim_leading_trailing
        self.groundtruth_sequences = SignificantFeaturesHelper.load_sequences(groundtruth_sequences_path, trim_leading_trailing)
        self.p_values = p_values
        self.k_values = k_values
        self.label = label
        self.compairr_path = compairr_path

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{RecoveredSignificantFeatures.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, RecoveredSignificantFeatures.__name__,
                                                            f"{RecoveredSignificantFeatures.__name__}/label")

        plotting_data = self._compute_plotting_data()
        table_result = self._write_results_table(plotting_data)

        fig_significant = self._safe_plot(plotting_data=plotting_data, column_of_interest="n_significant", y_label="Percentage of significant features that match the ground truth")
        fig_true = self._safe_plot(plotting_data=plotting_data, column_of_interest="n_true", y_label="Percentage of ground truth features that match the significant features")
        output_figures = [figure for figure in [fig_significant, fig_true] if figure]

        return ReportResult(name=self.name,
                            info="Compares a given collection of groundtruth implanted signals (sequences or k-mers) to the significant label-associated k-mers or sequences according to Fisher's exact test.",
                            output_figures=output_figures,
                            output_tables=[table_result])

    def _compute_plotting_data(self):
        result = {"encoding": [],
                  "p-value": [],
                  "n_significant": [],
                  "n_true": [],
                  "n_intersect": []}

        for k in self.k_values:
            encoder_name = SignificantFeaturesHelper._get_encoder_name(k)

            for p_value in self.p_values:
                significant_features = self._compute_significant_features(k, p_value)
                true_features = self._compute_true_features(k)

                result["encoding"].append(encoder_name)
                result["p-value"].append(p_value)
                result["n_significant"].append(len(significant_features))
                result["n_true"].append(len(true_features))
                result["n_intersect"].append(len(significant_features.intersection(true_features)))

        return pd.DataFrame(result)

    def _get_encoder_name(self, k):
        encoder_name = f"{k}-mer" if type(k) == int else k
        return encoder_name

    def _get_encoder_result_path(self, k, p_value):
        result_path = self.result_path / f"{self._get_encoder_name(k)}_{p_value}"
        PathBuilder.build(result_path)
        return result_path

    def _write_results_table(self, data) -> ReportOutput:
        table_path = self.result_path / f"recovered_significant_features_report.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "Number of true features found to be significant, and number of significant features found to be true.")

    def _plot(self, plotting_data, column_of_interest, y_label):
        plotting_data["percentage"] = plotting_data["n_intersect"] / plotting_data[column_of_interest]

        figure = px.bar(plotting_data, x="encoding", y="percentage", color=None,
                        facet_row=None, facet_col="p-value",
                        labels={
                            "percentage": y_label,
                            "encoding": "Encoding",
                            "class": "Repertoire class"
                        }, template='plotly_white',
                        color_discrete_sequence=px.colors.diverging.Tealrose,
                        range_y=[0, 1])

        figure.layout.yaxis.tickformat = ',.0%'

        file_path = self.result_path / f"{column_of_interest}_features_figure.html"

        figure.write_html(str(file_path))

        return ReportOutput(file_path, name=y_label)

    def _compute_significant_features(self, k, p_value):
        encoder_result_path = self._get_encoder_result_path(k, p_value)
        encoder_params = SignificantFeaturesHelper._build_encoder_params(self.label_config, encoder_result_path)

        if type(k) == int:
            significant_features = self._compute_significant_kmers(k, p_value, encoder_params)
        elif self.compairr_path is None:
            significant_features = self._compute_significant_sequences(p_value, encoder_params)
        else:
            significant_features = self._compute_significant_compairr_sequences(p_value, encoder_params)

        return set(significant_features)

    def _compute_significant_kmers(self, k, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_kmer_encoder(self.dataset, k, p_value, encoder_params)
        sequences = pd.read_csv(encoder.relevant_sequence_path)

        return list(sequences["k-mer"])

    def _compute_significant_sequences(self, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_sequence_encoder(self.dataset, p_value, encoder_params)
        sequences = pd.read_csv(encoder.relevant_sequence_path)

        return list(sequences[EnvironmentSettings.get_sequence_type().value])

    def _compute_significant_compairr_sequences(self, p_value, encoder_params):
        encoder = SignificantFeaturesHelper._build_compairr_sequence_encoder(self.dataset, p_value, encoder_params, self.compairr_path)
        sequences = pd.read_csv(encoder.relevant_sequence_path)

        return list(sequences[EnvironmentSettings.get_sequence_type().value])

    def _compute_true_features(self, k):
        if type(k) == int:
            return self._compute_true_kmers(k)
        else:
            return set(self.groundtruth_sequences)

    def _compute_true_kmers(self, k):
        kmers = set()

        for sequence in self.groundtruth_sequences:
            kmers = kmers.union(KmerHelper.create_kmers_from_string(sequence, k, overlap=True))

        return kmers

