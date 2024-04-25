import warnings
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SignificantFeaturesHelper import SignificantFeaturesHelper


class SequencesWithSignificantKmers(DataReport):
    """
    Given a list of reference sequences, this report writes out the subsets of reference sequences containing significant k-mers
    (as computed by the :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder` using Fisher's exact test).

    For each combination of p-value and k-mer size given, a file is written containing all sequences containing a significant
    k-mer of the given size at the given p-value.

    **Specification arguments:**

    - reference_sequences_path (str): Path to a file containing the reference sequences,
      The file should contain one sequence per line, without a header, and without V or J genes.

    - p_values (list): The p value thresholds to be used by Fisher's exact test. Each p-value specified here will become
      one panel in the output figure.

    - k_values (list): Length of the k-mers (number of amino acids) created by the
      :py:obj:`~immuneML.encodings.abundance_encoding.KmerAbundanceEncoder.KmerAbundanceEncoder`.
      Each k-mer length will become one panel in the output figure.

    - label (dict): A label configuration. One label should be specified, and the positive_class for this label should
      be defined. See the YAML specification below for an example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_sequences_with_significant_kmers:
                    SequencesWithSignificantKmers:
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
        location = SequencesWithSignificantKmers.__name__

        kwargs = SignificantFeaturesHelper.parse_parameters(kwargs, location)
        kwargs = SignificantFeaturesHelper.parse_sequences_path(kwargs, "reference_sequences_path", location)
        ParameterValidator.assert_all_type_and_value(kwargs["k_values"], int, location, "k_values")

        return SequencesWithSignificantKmers(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, reference_sequences_path: Path = None,
                 p_values: List[float] = None, k_values: List[int] = None, label: dict = None,
                 result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.reference_sequences_path = reference_sequences_path
        self.reference_sequences = SignificantFeaturesHelper.load_sequences(reference_sequences_path)
        self.p_values = p_values
        self.k_values = k_values
        self.label = label

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{SequencesWithSignificantKmers.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:
        self.label_config = LabelHelper.create_label_config([self.label], self.dataset, SequencesWithSignificantKmers.__name__,
                                                            f"{SequencesWithSignificantKmers.__name__}/label")

        report_outputs = self._write_output_files()

        return ReportResult(name=self.name,
                            info="Given a list of reference sequences, this report writes out the subsets of reference sequences containing significant k-mers.",
                            output_tables=report_outputs)

    def _write_output_files(self):
        report_outputs = []

        for k in self.k_values:
            for p_value in self.p_values:
                significant_kmers = self._compute_significant_kmers(k, p_value)
                output_file_path = self._get_output_file_path(k, p_value)
                self._write_sequences_containing_significant_kmers(significant_kmers, output_file_path)

                report_outputs.append(ReportOutput(output_file_path,
                                                   f"Sequences containing significant {k}-mers with p-value {p_value}"))
        return report_outputs

    def _get_encoder_result_path(self, k, p_value):
        result_path = self.result_path / f"{k}-mer_{p_value}"
        PathBuilder.build(result_path)
        return result_path

    def _get_output_file_path(self, k, p_value):
        return self.result_path / f"sequences_with_significant_{k}-mers_at_p={p_value}.txt"

    def _write_sequences_containing_significant_kmers(self, significant_kmers, output_file):
        with open(output_file, "w") as f:
            for sequence in self.reference_sequences:
                for kmer in significant_kmers:
                    if kmer in sequence:
                        f.write(sequence)
                        f.write("\n")
                        break

        f.close()

    def _compute_significant_kmers(self, k, p_value):
        encoder_result_path = self._get_encoder_result_path(k, p_value)
        encoder_params = SignificantFeaturesHelper._build_encoder_params(self.label_config, encoder_result_path)
        encoder = SignificantFeaturesHelper._build_kmer_encoder(self.dataset, k, p_value, encoder_params)
        sequences = pd.read_csv(encoder.relevant_sequence_path)

        return list(sequences["k-mer"])