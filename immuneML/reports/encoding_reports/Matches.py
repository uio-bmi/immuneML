import itertools
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class Matches(EncodingReport):
    """
    Reports the number of matches that were found when using one of the following encoders:

    * :ref:`MatchedSequences` encoder
    * :ref:`MatchedReceptors` encoder
    * :ref:`MatchedRegex` encoder


    Report results are:

    * A table containing all matches, where the rows correspond to the Repertoires, and the
      columns correspond to the objects to match (regular expressions or receptor sequences).
    * The repertoire sizes (read frequencies and the number of unique sequences per repertoire), for each of the chains.
      This can be used to calculate the percentage of matched sequences in a repertoire.
    * When using :ref:`MatchedSequences` encoder or
      :ref:`MatchedReceptors` encoder, tables describing
      the chains and receptors (ids, chains, V and J genes and sequences).
    * When using :ref:`MatchedReceptors` encoder or using
      :ref:`MatchedRegex` encoder with chain pairs, tables describing
      the paired matches (where a match was found in both chains) per repertoire.


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_match_report: Matches
    """

    @classmethod
    def build_object(cls, **kwargs):
        return Matches(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path
        self.name = name

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        return self._write_reports()

    def _write_reports(self) -> ReportResult:
        all_matches_table = self._write_match_table()
        repertoire_sizes = self._write_repertoire_sizes()

        output_tables = [all_matches_table, repertoire_sizes]

        if self.dataset.encoded_data.encoding == "MatchedSequencesEncoder":
            output_tables += self._write_sequence_info(self.result_path / "sequence_info")
        else:
            if len(self.dataset.encoded_data.feature_annotations["chain"].unique()) == 2:
                output_tables += self._write_paired_matches(self.result_path / "paired_matches")

            if self.dataset.encoded_data.encoding == "MatchedReceptorsEncoder":
                output_tables += self._write_receptor_info(self.result_path / "receptor_info")

        return ReportResult(self.name, output_tables=output_tables)

    def _write_match_table(self):
        id_df = pd.DataFrame({"repertoire_id": self.dataset.encoded_data.example_ids})
        label_df = pd.DataFrame(self.dataset.encoded_data.labels)
        matches_df = pd.DataFrame(self.dataset.encoded_data.examples, columns=self.dataset.encoded_data.feature_names)

        result_path = self.result_path / "complete_match_count_table.csv"
        id_df.join(label_df).join(matches_df).to_csv(result_path, index=False)

        return ReportOutput(result_path, "All matches")

    def _write_paired_matches(self, paired_matches_path: Path) -> List[ReportOutput]:
        PathBuilder.build(paired_matches_path)

        report_outputs = []
        for i in range(0, len(self.dataset.encoded_data.example_ids)): # todo don't mention subject in the name twice
            file_name = "example_{}_".format(self.dataset.encoded_data.example_ids[i])
            file_name += "_".join(["{label}_{value}".format(label=label, value=values[i]) for
                                  label, values in self.dataset.encoded_data.labels.items()])
            file_name += ".csv"
            file_path = paired_matches_path / file_name

            if self.dataset.encoded_data.encoding == "MatchedReceptorsEncoder":
                self._write_paired_receptor_matches_for_repertoire(self.dataset.encoded_data.examples[i], file_path)
            elif self.dataset.encoded_data.encoding == "MatchedRegexEncoder":
                self._write_paired_regex_matches_for_repertoire(self.dataset.encoded_data.examples[i], file_path)

            report_outputs.append(ReportOutput(file_path, f"Example {self.dataset.encoded_data.example_ids[i]} paired matches"))

        return report_outputs

    def _write_paired_receptor_matches_for_repertoire(self, matches, filename):
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

    def _write_paired_regex_matches_for_repertoire(self, matches, filename):
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

    def _write_repertoire_sizes(self):
        """
        Writes the repertoire sizes (# clones & # reads) per subject, per chain.
        """
        all_subjects = self.dataset.encoded_data.example_ids
        all_chains = sorted(set(self.dataset.encoded_data.feature_annotations["chain"]))

        results_df = pd.DataFrame(list(itertools.product(all_subjects, all_chains)),
                                  columns=["subject_id", "chain"])
        results_df["n_reads"] = 0
        results_df["n_clones"] = 0

        for repertoire in self.dataset.repertoires:
            rep_counts = repertoire.get_counts()
            rep_chains = repertoire.get_chains()

            for chain in all_chains:
                indices = rep_chains == Chain.get_chain(chain.upper())
                results_df.loc[(results_df.subject_id == repertoire.metadata["subject_id"]) & (results_df.chain == chain),
                               'n_reads'] += np.sum(rep_counts[indices])
                results_df.loc[(results_df.subject_id == repertoire.metadata["subject_id"]) & (results_df.chain == chain),
                               'n_clones'] += len(rep_counts[indices])

        results_path = self.result_path / "repertoire_sizes.csv"
        results_df.to_csv(results_path, index=False)

        return ReportOutput(results_path, "Repertoire sizes")

    def _write_receptor_info(self, receptor_info_path) -> List[ReportOutput]:
        PathBuilder.build(receptor_info_path)

        receptor_chains = self.dataset.encoded_data.feature_annotations
        chain_types = receptor_chains["chain"].unique()

        first_chains = receptor_chains.loc[receptor_chains.chain == chain_types[0]]
        second_chains = receptor_chains.loc[receptor_chains.chain == chain_types[1]]

        first_chains.drop(columns=["chain"], inplace=True)
        second_chains.drop(columns=["chain"], inplace=True)

        on_cols = ["receptor_id"]
        if "clonotype_id" in second_chains.columns and first_chains.columns:
            on_cols += ["clonotype_id"]

        receptors = pd.merge(first_chains, second_chains,
                             on=on_cols,
                             suffixes=(f"_{chain_types[0]}", f"_{chain_types[1]}"))

        unique_alpha_chains = first_chains.drop_duplicates(subset=["sequence", "v_gene", "j_gene"])
        unique_beta_chains = second_chains.drop_duplicates(subset=["sequence", "v_gene", "j_gene"])
        unique_receptors = receptors.drop_duplicates(subset=[f"sequence_{chain_types[0]}", f"v_gene_{chain_types[0]}", f"j_gene_{chain_types[0]}",
                                                             f"sequence_{chain_types[1]}", f"v_gene_{chain_types[1]}", f"j_gene_{chain_types[1]}"])

        receptor_chains_path = receptor_info_path / "all_chains.csv"
        receptor_chains.to_csv(receptor_chains_path, index=False)
        receptors_path = receptor_info_path / "all_receptors.csv"
        receptors.to_csv(receptors_path, index=False)
        unique_chain1_path = receptor_info_path / f"unique_{chain_types[0]}_chains.csv"
        unique_alpha_chains.to_csv(unique_chain1_path, index=False)
        unique_chain2_path = receptor_info_path / f"unique_{chain_types[1]}_chains.csv"
        unique_beta_chains.to_csv(unique_chain2_path, index=False)
        unique_receptors_path = receptor_info_path / "unique_receptors.csv"
        unique_receptors.to_csv(unique_receptors_path, index=False)

        return [ReportOutput(path=path, name=name) for path, name in [(receptors_path, "All receptors info"),
                                                                      (receptor_chains_path, "All receptor chains info"),
                                                                      (unique_receptors_path, "Unique receptors info"),
                                                                      (unique_chain1_path, "Unique chain 1 info"),
                                                                      (unique_chain2_path, "Unique chain 2 info")]]

    def _write_sequence_info(self, sequence_info_path) -> List[ReportOutput]:
        PathBuilder.build(sequence_info_path)

        chains = self.dataset.encoded_data.feature_annotations
        unique_chains = chains.drop_duplicates(subset=["sequence", "v_gene", "j_gene"])

        chains_path = sequence_info_path / "all_chains.csv"
        chains.to_csv(chains_path, index=False)
        unique_chains_path = sequence_info_path / "unique_chains.csv"
        unique_chains.to_csv(unique_chains_path, index=False)

        return [ReportOutput(path=path, name=name) for path, name in [(chains_path, "All chains info"), (unique_chains_path, "Unique chains info")]]

    def check_prerequisites(self):
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn(f"No encoding was specified for dataset {self.dataset.identifier}. Please use one of the following encodings: MatchedReceptorsEncoder, MatchedSequencesEncoder, MatchedRegexEncoder. Matches report will not be created.")
            return False

        if self.dataset.encoded_data.encoding not in ("MatchedReceptorsEncoder", "MatchedSequencesEncoder", "MatchedRegexEncoder"):
            warnings.warn(f"Encoding {self.dataset.encoded_data.encoding} is not compatible with this report type. Matches report will not be created.")
            return False
        else:
            return True