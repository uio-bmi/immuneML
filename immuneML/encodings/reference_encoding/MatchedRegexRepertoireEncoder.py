import re
import logging

import numpy as np
import pandas as pd

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
from immuneML.util.Logger import print_log
from immuneML.util.ReadsType import ReadsType


class MatchedRegexRepertoireEncoder(MatchedRegexEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        self._load_regex_df()

        feature_annotations = self._get_feature_info()
        encoded_repertoires, labels = self._encode_repertoires(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=encoded_repertoires,
            example_ids=dataset.get_example_ids(),
            feature_names=list(feature_annotations["locus_id"]),
            feature_annotations=feature_annotations,
            labels=labels,
            encoding=MatchedRegexEncoder.__name__,
            info={'sequence_type': params.sequence_type,
                  'region_type': params.region_type}
        )

        return encoded_dataset

    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - feature id (id_CHAIN)
         - regex
         - v_gene (if match_v_genes == True)
        only for the motifs for which a regex was specified
        """
        features = {"regex_id": [], "locus_id": [], "locus": [], "regex": []}

        if self.match_v_genes:
            features["v_call"] = []

        for index, row in self.regex_df.iterrows():
            for chain_type in self.chains:
                regex = row[f"{chain_type}_regex"]

                if regex is not None:
                    features["regex_id"].append(f"{row['id']}")
                    features["locus_id"].append(f"{row['id']}_{chain_type}")
                    features["locus"].append(Chain.get_chain(chain_type).name.lower())
                    features["regex"].append(regex)

                    if self.match_v_genes:
                        v_gene = row[f"{chain_type}V"] if f"{chain_type}V" in row else None
                        features["v_call"].append(v_gene)

        return pd.DataFrame(features)

    def _encode_repertoires(self, dataset: RepertoireDataset, params: EncoderParams):
        # Rows = repertoires, Columns = regex matches (one chain per column)
        encoded_repertoires = np.zeros((dataset.get_example_count(), self.feature_count), dtype=int)
        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        n_repertoires = dataset.get_example_count()

        for i, repertoire in enumerate(dataset.get_data()):
            print_log(f"Encoding repertoire {i+1}/{n_repertoires}", include_datetime=True)
            encoded_repertoires[i] = self._match_repertoire_to_regexes(repertoire, params)

            if labels is not None:
                for label_name in params.label_config.get_labels_by_name():
                    labels[label_name].append(repertoire.metadata[label_name])

        return encoded_repertoires, labels

    def _match_repertoire_to_regexes(self, repertoire: Repertoire, params: EncoderParams):
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences(params.region_type)

        match_idx = 0

        for index, row in self.regex_df.iterrows():
            for chain_type in self.chains:
                regex = row[f"{chain_type}_regex"]

                if regex is not None:
                    v_gene = row[f"{chain_type}V"] if f"{chain_type}V" in row else None

                    for rep_seq in rep_seqs:
                        if rep_seq.locus is not None:
                            if rep_seq.locus == chain_type:
                                if self._matches(rep_seq, regex, v_gene):
                                    n_matches = 1 if self.reads == ReadsType.UNIQUE else rep_seq.duplicate_count
                                    if n_matches is None:
                                        logging.warning(f"MatchedRegexRepertoireEncoder: count not defined for sequence with id {rep_seq.sequence_id} in repertoire {repertoire.identifier}, ignoring sequence...")
                                        n_matches = 0
                                    matches[match_idx] += n_matches
                        else:
                            logging.warning(f"{MatchedRegexRepertoireEncoder.__class__.__name__}: chain was not set for sequence {rep_seq.sequence_id}, skipping the sequence for matching...")
                    match_idx += 1

        return matches

    def _matches(self, receptor_sequence, regex, v_gene=None):
        if v_gene is not None and receptor_sequence.v_call != v_gene:
            matches = False
        else:
            matches = bool(re.search(regex, receptor_sequence.sequence_aa))

        return matches
