import itertools
import os
import warnings
from typing import List

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class RegexMatches(EncodingReport):
    """
    todo fill this in
    Specification:
    .. indent with spaces
    .. code-block:: yaml
        my_match_report: RegexMatches
    """

    @classmethod
    def build_object(cls, **kwargs):
        return RegexMatches(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.name = name

    def generate(self) -> ReportResult:
        PathBuilder.build(os.path.join(self.result_path, "paired_matches"))
        return self._write_reports()

    def _write_reports(self) -> ReportResult:
        all_chain_matches_table = self._write_match_table()
        paired_matches_list = self._write_paired_matches()
        # repertoire_sizes = self._write_repertoire_sizes()

        return ReportResult(self.name, output_tables=[all_chain_matches_table] + paired_matches_list)

    def _write_match_table(self):
        id_df = pd.DataFrame({"repertoire_id": self.dataset.encoded_data.example_ids})
        label_df = pd.DataFrame(self.dataset.encoded_data.labels)
        matches_df = pd.DataFrame(self.dataset.encoded_data.examples, columns=self.dataset.encoded_data.feature_names)

        result_path = os.path.join(self.result_path, "complete_match_count_table.csv")
        id_df.join(label_df).join(matches_df).to_csv(result_path, index=False)

        return ReportOutput(result_path, "All paired matches")

    def _write_paired_matches(self) -> List[ReportOutput]:
        report_outputs = []
        for i in range(0, len(self.dataset.encoded_data.example_ids)):
            filename = "example_{}_".format(self.dataset.encoded_data.example_ids[i])
            filename += "_".join(["{label}_{value}".format(label=label, value=values[i]) for
                                  label, values in self.dataset.encoded_data.labels.items()])
            filename += ".csv"
            filename = os.path.join(self.result_path, "paired_matches", filename)

            self._write_paired_matches_for_repertoire(self.dataset.encoded_data.examples[i],
                                                      filename)
            report_outputs.append(ReportOutput(filename, f"example {self.dataset.encoded_data.example_ids[i]} paired matches"))

        return report_outputs

    def _write_paired_matches_for_repertoire(self, matches, filename):
        match_identifiers = []
        match_values = []

        annotation_df = self.dataset.encoded_data.feature_annotations

        for receptor_id in sorted(set(annotation_df["receptor_id"])):
            chain_ids = list(annotation_df.loc[annotation_df["receptor_id"] == receptor_id]["chain_id"])

            if len(chain_ids) == 2:
                first_match_idx = self.dataset.encoded_data.feature_names.index(chain_ids[0])
                second_match_idx = self.dataset.encoded_data.feature_names.index(chain_ids[1])

                if matches[first_match_idx] > 0 and matches[second_match_idx] > 0:
                    match_identifiers.append(chain_ids[0])
                    match_identifiers.append(chain_ids[1])
                    match_values.append(matches[first_match_idx])
                    match_values.append(matches[second_match_idx])

        results_df = pd.DataFrame([match_values], columns=match_identifiers)
        results_df.to_csv(filename, index=False)

    # def _write_repertoire_sizes(self):
    #     """
    #     Writes the repertoire sizes (# clones & # reads) per subject, per chain.
    #     """
    #     all_subjects = self.dataset.encoded_data.example_ids
    #     all_chains = sorted(set(self.dataset.encoded_data.feature_annotations["chain"]))
    #
    #     results_df = pd.DataFrame(list(itertools.product(all_subjects, all_chains)),
    #                               columns=["subject_id", "chain"])
    #     results_df["n_reads"] = 0
    #     results_df["n_clones"] = 0
    #
    #     for repertoire in self.dataset.repertoires:
    #         rep_counts = repertoire.get_counts()
    #         rep_chains = repertoire.get_attribute("chains")
    #
    #         for chain in all_chains:
    #             chain_enum = Chain.get_chain(chain[0].upper())
    #             indices = rep_chains == chain_enum
    #             results_df.loc[(results_df.subject_id == repertoire.metadata["subject_id"]) & (results_df.chain == chain),
    #                            'n_reads'] += np.sum(rep_counts[indices])
    #             results_df.loc[(results_df.subject_id == repertoire.metadata["subject_id"]) & (results_df.chain == chain),
    #                            'n_clones'] += len(rep_counts[indices])
    #
    #     results_path = os.path.join(self.result_path, "repertoire_sizes.csv")
    #     results_df.to_csv(results_path, index=False)
    #
    #     return ReportOutput(results_path, "repertoire sizes")

    def check_prerequisites(self):
        if "MatchedRegexEncoder" != self.dataset.encoded_data.encoding:
            warnings.warn("Encoding is not compatible with the report type. PairedReceptorMatches report will not be created.")
            return False
        else:
            return True