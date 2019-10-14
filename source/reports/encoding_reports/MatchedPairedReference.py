import os
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class MatchedPairedReference(EncodingReport):
    """
    Uses MatchedReceptorsEncoder to encode the dataset, and reports the number of matches between
    a dataset containing unpaired immune receptor sequences, and a set of paired reference receptors. 
    """

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None):
        self.dataset = dataset
        self.result_path = result_path

    def generate(self):
        PathBuilder.build(os.path.join(self.result_path, "paired_matches"))
        self._write_reports()

    def _write_reports(self):
        self._write_results_table_all_chains()
        self._write_receptor_sequence_details_report()
        self._write_paired_matches()

    def _write_results_table_all_chains(self):
        id_df = pd.DataFrame({"repertoire_id": self.dataset.encoded_data.example_ids})
        label_df = pd.DataFrame(self.dataset.encoded_data.labels)
        matches_df = pd.DataFrame(self.dataset.encoded_data.examples, columns=self.dataset.encoded_data.feature_names)

        id_df.join(label_df).join(matches_df).to_csv(os.path.join(self.result_path, "complete_match_count_table.csv"), index=False)

    def _write_receptor_sequence_details_report(self):
        self.dataset.encoded_data.feature_annotations.to_csv(os.path.join(self.result_path, "receptor_sequence_details.csv"), index=False)

    def _write_paired_matches(self):
        for i in range(0, len(self.dataset.encoded_data.example_ids)):
            filename = "Dataset={}_".format(self.dataset.encoded_data.example_ids[i])
            filename += ",".join(["{label}={value}".format(label=label, value=values[i]) for
                                                       label, values in self.dataset.encoded_data.labels.items()])
            filename += ".csv"
            filename = os.path.join(self.result_path, "paired_matches", filename)

            self._write_paired_matches_for_repertoire(self.dataset.encoded_data.examples[i],
                                                      filename)

    def _write_paired_matches_for_repertoire(self, matches, filename):
        match_identifiers = []
        match_values = []

        for i in range(0, int(len(matches) / 2)):
            first_match_idx = i * 2
            second_match_idx = i * 2 + 1

            if matches[first_match_idx] > 0 and matches[second_match_idx] > 0:
                match_identifiers.append(self.dataset.encoded_data.feature_names[first_match_idx])
                match_identifiers.append(self.dataset.encoded_data.feature_names[second_match_idx])
                match_values.append(matches[first_match_idx])
                match_values.append(matches[second_match_idx])

        results_df = pd.DataFrame([match_values], columns=match_identifiers)
        results_df.to_csv(filename, index=False)

    def check_prerequisites(self):
        assert "MatchedReceptorsRepertoireEncoder" == self.dataset.encoded_data.encoding, "Encoding is not compatible with the report type. " \
                                                                                          "MatchedPairedReference report will not be created."
