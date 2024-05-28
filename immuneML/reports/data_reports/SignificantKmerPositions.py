import warnings
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.KmerHelper import KmerHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SignificantFeaturesHelper import SignificantFeaturesHelper


class SignificantKmerPositions(DataReport):
    """
    Plots the number of significant k-mers (as computed by the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder` using Fisher's exact test)
    observed at each IMGT position of a given list of reference sequences.
    This report creates a stacked bar chart, where each bar represents an IMGT position, and each segment of the stack represents the observed frequency
    of one 'significant' k-mer at that position.

    **Specification arguments:**

    - reference_sequences_path (str): Path to a file containing the reference sequences,
      The file should contain one sequence per line, without a header, and without V or J genes.

    - p_values (list): The p value thresholds to be used by Fisher's exact test. Each p-value specified here will become one panel in the output figure.

    - k_values (list): Length of the k-mers (number of amino acids) created by the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`.
      Each k-mer length will become one panel in the output figure.

    - label (dict): A label configuration. One label should be specified, and the positive_class for this label should be defined. See the YAML specification below for an example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_significant_kmer_positions_report:
                    SignificantKmerPositions:
                        reference_sequences_path: path/to/reference/sequences.txt
                        p_values:
                            - 0.1
                            - 0.01
                            - 0.001
                            - 0.0001
                        k_values:
                            - 3
                            - 4
                            - 5
                        label: # Define a label, and the positive class for that given label
                            CMV:
                                positive_class: +
    """

    @classmethod
    def build_object(cls, **kwargs):
        location = SignificantKmerPositions.__name__

        kwargs = SignificantFeaturesHelper.parse_parameters(kwargs, location)
        kwargs = SignificantFeaturesHelper.parse_sequences_path(kwargs, "reference_sequences_path", location)
        ParameterValidator.assert_all_type_and_value(kwargs["k_values"], int, location, "k_values")

        return SignificantKmerPositions(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, reference_sequences_path: Path = None,
                 p_values: List[float] = None, k_values: List[int] = None, label: dict = None,
                 compairr_path: Path = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.reference_sequences_path = reference_sequences_path
        self.reference_sequences = SignificantFeaturesHelper.load_sequences(reference_sequences_path)
        self.p_values = p_values
        self.k_values = k_values
        self.label = label
        self.compairr_path = compairr_path
        self.label_config = None

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{SignificantKmerPositions.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, SignificantKmerPositions.__name__,
                                                            f"{SignificantKmerPositions.__name__}/label")

        plotting_data = self._compute_plotting_data()
        table_result = self._write_results_table(plotting_data)

        report_output_fig = self._safe_plot(plotting_data=plotting_data)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(name=self.name,
                            info="The number of significant k-mers observed at each IMGT position of a given list of reference sequences.",
                            output_figures=output_figures,
                            output_tables=[table_result])

    def _compute_plotting_data(self):
        result = {"encoding": [],
                  "p-value": [],
                  "imgt_position": [],
                  "k-mer": [],
                  "count": []}

        for k in self.k_values:
            for p_value in self.p_values:
                significant_kmer_positions = self._compute_significant_kmer_positions(k, p_value)

                for imgt_pos, kmer_dict in significant_kmer_positions.items():
                    for kmer, count in kmer_dict.items():
                        result["encoding"].append(f"{k}-mer")
                        result["p-value"].append(p_value)
                        result["imgt_position"].append(str(imgt_pos))
                        result["k-mer"].append(kmer)
                        result["count"].append(count)

        return pd.DataFrame(result).astype({'imgt_position': str})

    def _get_encoder_result_path(self, k, p_value):
        result_path =  self.result_path / f"{k}-mer_{p_value}"
        PathBuilder.build(result_path)
        return result_path

    def _write_results_table(self, data) -> ReportOutput:
        table_path = self.result_path / f"significant_kmer_positions_report.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, "Number of significant k-mers found at each position in a set of reference sequences")

    def _plot(self, plotting_data):
        figure = px.bar(plotting_data, x="imgt_position", y="count", color="k-mer",
                        facet_row="encoding", facet_col="p-value",
                        labels={
                            "encoding": "Encoding",
                            "imgt_position": "sequence position (IMGT scheme)",
                            "count": "Number of significant k-mers observed"
                        }, template="plotly_white",
                        category_orders={
                            "imgt_position": self._get_imgt_position_order(set(plotting_data["imgt_position"]))
                        },
                        barmode="stack",
                        color_discrete_sequence=px.colors.diverging.Tealrose)

        file_path = self.result_path / f"significant_kmer_positions.html"

        figure.write_html(str(file_path))

        return ReportOutput(file_path, name="Significant k-mers observed at each position in the reference sequences")

    def _get_imgt_position_order(self, imgt_positions):
        sorted_positions = sorted([float(pos) for pos in imgt_positions])
        return [str(pos_float) if int(pos_float) != pos_float else str(int(pos_float)) for pos_float in sorted_positions]

    def _compute_significant_kmer_positions(self, k, p_value):
        significant_kmers = self._compute_significant_kmers(k, p_value)

        results = {}

        for sequence in self.reference_sequences:
            reference_imgt_kmers = KmerHelper.create_IMGT_kmers_from_string(sequence, k, region_type=RegionType.IMGT_CDR3)

            for kmer, imgt_pos in reference_imgt_kmers:
                if imgt_pos not in results:
                    results[imgt_pos] = {}

                if kmer in significant_kmers:
                    if kmer in results[imgt_pos]:
                        results[imgt_pos][kmer] += 1
                    else:
                        results[imgt_pos][kmer] = 1

        return results

    def _compute_significant_kmers(self, k, p_value):
        encoder_result_path = self._get_encoder_result_path(k, p_value)
        encoder_params = SignificantFeaturesHelper._build_encoder_params(self.label_config, encoder_result_path)
        encoder = SignificantFeaturesHelper._build_kmer_encoder(self.dataset, k, p_value, encoder_params)
        sequences = pd.read_csv(encoder.relevant_sequence_path)

        return list(sequences["k-mer"])