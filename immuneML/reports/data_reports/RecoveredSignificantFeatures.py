import warnings
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import os

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.KmerHelper import KmerHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SignificantFeaturesHelper import SignificantFeaturesHelper


class RecoveredSignificantFeatures(DataReport):
    """
    xxx

    Internally uses the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder` for calculating
    significant k-mers, and
    :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
    :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
    to calculate significant full sequences (depending on whether the argument compairr_path was set).

    Arguments:

        groundtruth_sequences_path (str): Path to a file containing the true implanted (sub)sequences, e.g., full sequences or k-mers.
        The file should contain one sequence per line, without a header, and without V or J genes.

        p_values (list): The p value thresholds to be used by Fisher's exact test. Each p-value specified here will become one panel in the output figure.

        k_values (list): Length of the k-mers (number of amino acids) created by the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`.
        When using a full sequence encoding (:py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
        :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`), specify 'full_sequence' here.
        Each value specified under k_values will represent one boxplot in the output figure.

        label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.

        compairr_path (str): If 'full_sequence' is listed under k_values, the path to the CompAIRR executable may be provided.
        If the compairr_path is specified, the :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder`
        will be used to compute the significant sequences. If the path is not specified and 'full_sequence' is listed under
        k-values, :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` will be used.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_significant_features_report:
            RecoveredSignificantFeatures:
                groundtruth_sequences_path: path/to/groundtruth/sequences.txt
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

        ParameterValidator.assert_keys_present(kwargs.keys(), ["groundtruth_sequences_path"], location, location)
        ParameterValidator.assert_type_and_value(kwargs["groundtruth_sequences_path"], str, location, "groundtruth_sequences_path")
        assert os.path.isfile(kwargs["groundtruth_sequences_path"]), f"{location}: implanted_sequences_path does not exist: {kwargs['groundtruth_sequences_path']}"

        kwargs["groundtruth_sequences_path"] = Path(kwargs["groundtruth_sequences_path"])

        kwargs = SignificantFeaturesHelper.parse_parameters(kwargs, location)

        return RecoveredSignificantFeatures(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, groundtruth_sequences_path: Path = None,
                 p_values: List[float] = None, k_values: List[int] = None, label_config: LabelConfiguration = None,
                 compairr_path: Path = None, result_path: Path = None, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.groundtruth_sequences_path = groundtruth_sequences_path
        self.groundtruth_sequences = self._load_groundtruth_sequences(groundtruth_sequences_path)
        self.p_values = p_values
        self.k_values = k_values
        self.label_config = label_config
        self.compairr_path = compairr_path

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{RecoveredSignificantFeatures.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _load_groundtruth_sequences(self, groundtruth_sequences_path):
        with open(groundtruth_sequences_path) as f:
            readlines = f.readlines()
            sequences = [seq.strip() for seq in readlines]
        return sequences

    def _generate(self) -> ReportResult:
        plotting_data = self._compute_plotting_data()
        table_result = self._write_results_table(plotting_data)

        fig_significant = self._safe_plot(plotting_data=plotting_data, column_of_interest="n_significant", y_label="Percentage of significant features that match the ground truth")
        fig_true = self._safe_plot(plotting_data=plotting_data, column_of_interest="n_true", y_label="Percentage of ground truth features that match the significant features")
        output_figures = [figure for figure in [fig_significant, fig_true] if figure]

        return ReportResult(self.name, output_figures, [table_result])

    def _compute_plotting_data(self):
        result = {"encoding": [],
                  "p-value": [],
                  "n_significant": [],
                  "n_true": [],
                  "n_intersect": []}

        for k in self.k_values:
            encoder_name = SignificantFeaturesHelper._get_encoder_name(k)

            for p_value in self.p_values:
                encoder_result_path = self._get_encoder_result_path(k, p_value)
                significant_features = self._compute_significant_features(k, p_value, encoder_result_path)
                true_features = self._compute_true_features(k)

                result["encoding"].append(encoder_name)
                result["p-value"].append(p_value)
                result["n_significant"].append(len(significant_features))
                result["n_true"].append(len(true_features))
                result["n_intersect"].append(len(significant_features.intersection(true_features)))

        return pd.DataFrame(result)

    def _get_positive_negative_classes(self):
        label = self.label_config.get_label_objects()[0]
        positive_class = label.positive_class
        negative_class = [value for value in label.values if value != positive_class][0]

        return positive_class, negative_class

    def _get_encoder_name(self, k):
        encoder_name = f"{k}-mer" if type(k) == int else k
        return encoder_name

    def _get_encoder_result_path(self, k, p_value):
        return self.result_path / f"{self._get_encoder_name(k)}_{p_value}"

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

    def _compute_significant_features(self, k, p_value, encoder_result_path):
        PathBuilder.build(encoder_result_path)

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

