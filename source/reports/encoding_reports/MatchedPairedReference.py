import os
import pandas as pd
import numpy as np
import warnings
import itertools

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class MatchedPairedReference(EncodingReport):
    """
    Reports the number of matches between a dataset containing unpaired (single chain) immune receptor
    sequences, and a set of paired reference receptors.
    :py:obj:`~source.encodings.reference_encoding.MatchedReceptorsEncoder.MatchedReceptorsEncoder`
    must be used to encode the dataset.


    Specification:

        definitions:
            datasets:
                unpaired_data:
                    ...
            encodings:
                my_mr_encoding:
                    MatchedReceptors:
                        ...
            reports:
                my_mr_report:
                    MatchedPairedReference
        instructions:
                instruction_1:
                    type: ExploratoryAnalysis
                    analyses:
                        my_mr_analysis:
                            dataset: unpaired_data
                            encoding: my_mr_encoding
                            report: my_mr_report
                            labels:
                                - ...
    """

    @classmethod
    def build_object(cls, **kwargs):
        return MatchedPairedReference(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None):
        self.dataset = dataset
        self.result_path = result_path

    def generate(self):
        PathBuilder.build(os.path.join(self.result_path, "paired_matches"))
        PathBuilder.build(os.path.join(self.result_path, "receptor_info"))
        self._write_reports()

    def _write_reports(self):
        self._write_results_table_all_chains()
        self._write_paired_matches()
        self._write_receptor_info()
        self._write_repertoire_sizes()

    def _write_results_table_all_chains(self):
        id_df = pd.DataFrame({"repertoire_id": self.dataset.encoded_data.example_ids})
        label_df = pd.DataFrame(self.dataset.encoded_data.labels)
        matches_df = pd.DataFrame(self.dataset.encoded_data.examples, columns=self.dataset.encoded_data.feature_names)

        id_df.join(label_df).join(matches_df).to_csv(os.path.join(self.result_path, "complete_match_count_table.csv"), index=False)

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

    def _write_repertoire_sizes(self):
        """
        Writes the repertoire sizes (# clones & # reads) per donor, per chain.
        """
        all_donors = self.dataset.encoded_data.example_ids
        all_chains = sorted(set(self.dataset.encoded_data.feature_annotations["chain"]))

        results_df = pd.DataFrame(list(itertools.product(all_donors, all_chains)),
                          columns=["donor_id", "chain"])
        results_df["n_reads"] = 0
        results_df["n_clones"] = 0

        for repertoire in self.dataset.repertoires:
            rep_counts = repertoire.get_attribute("counts")
            rep_chains = repertoire.get_attribute("chains")

            for chain in all_chains:
                chain_enum = Chain(chain[0].upper())
                indices = rep_chains == chain_enum
                results_df.loc[(results_df.donor_id == repertoire.metadata["donor"]) & (results_df.chain == chain),
                       'n_reads'] += np.sum(rep_counts[indices])
                results_df.loc[(results_df.donor_id == repertoire.metadata["donor"]) & (results_df.chain == chain),
                       'n_clones'] += len(rep_counts[indices])

        results_df.to_csv(os.path.join(self.result_path, "repertoire_sizes.csv"), index=False)

    def _write_receptor_info(self):
        receptor_info_path = os.path.join(self.result_path, "receptor_info")

        receptor_chains = self.dataset.encoded_data.feature_annotations

        alpha_chains = receptor_chains.loc[receptor_chains.chain=="alpha"]
        beta_chains = receptor_chains.loc[receptor_chains.chain=="beta"]

        alpha_chains.drop(columns=["chain"], inplace=True)
        beta_chains.drop(columns=["chain"], inplace=True)

        receptors = pd.merge(alpha_chains, beta_chains,
                             on=["receptor_id", "clonotype_id"],
                             suffixes=("_alpha", "_beta"))

        unique_alpha_chains = alpha_chains.drop_duplicates(subset=["sequence", "v_gene", "j_gene"])
        unique_beta_chains = beta_chains.drop_duplicates(subset=["sequence", "v_gene", "j_gene"])
        unique_receptors = receptors.drop_duplicates(subset=["sequence_alpha", "v_gene_alpha", "j_gene_alpha",
                                                             "sequence_beta", "v_gene_beta", "j_gene_beta"])

        receptor_chains.to_csv(os.path.join(receptor_info_path, "all_chains.csv"), index=False)
        receptors.to_csv(os.path.join(receptor_info_path, "all_receptors.csv"), index=False)
        unique_alpha_chains.to_csv(os.path.join(receptor_info_path, "unique_alpha_chains.csv"), index=False)
        unique_beta_chains.to_csv(os.path.join(receptor_info_path, "unique_beta_chains.csv"), index=False)
        unique_receptors.to_csv(os.path.join(receptor_info_path, "unique_receptors.csv"), index=False)



    def check_prerequisites(self):
        if "MatchedReceptorsRepertoireEncoder" != self.dataset.encoded_data.encoding:
            warnings.warn("Encoding is not compatible with the report type. MatchedPairedReference report will not be created.")
            return False
        else:
            return True

