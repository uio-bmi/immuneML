import numpy as np
import pandas as pd
import re
import datetime

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder


class MatchedRegexRepertoireEncoder(MatchedRegexEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        self._load_regex_df()

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, params=dataset.params,
                                            metadata_file=dataset.metadata_file)

        feature_annotations = self._get_feature_info()
        encoded_repertoires, labels = self._encode_repertoires(dataset, params)

        encoded_dataset.add_encoded_data(EncodedData(
            examples=encoded_repertoires,
            example_ids=labels["subject_id"], # todo label subject_id must always be specified? should this automatically be added?
            feature_names=list(feature_annotations["chain_id"]),
            feature_annotations=feature_annotations,
            labels=labels,
            encoding=MatchedRegexEncoder.__name__
        ))

        self.store(encoded_dataset, params)
        return encoded_dataset


    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - feature id (id_CHAIN)
         - regex
         - v_gene (if match_v_genes == True)
        only for the motifs for which a regex was specified
        """
        features = {"receptor_id": [], "chain_id": [], "chain": [], "regex": []}

        if self.match_v_genes:
            features["v_gene"] = []

        for index, row in self.regex_df.iterrows():
            for chain_type in self.chains: # todo generalize to other chain types!
                regex = row[f"{chain_type}_regex"]

                if regex is not None:
                    features["receptor_id"].append(f"{row['id']}")
                    features["chain_id"].append(f"{row['id']}_{chain_type}")
                    features["chain"].append(Chain.get_chain(chain_type).name.lower())
                    features["regex"].append(regex)

                    if self.match_v_genes:
                        v_gene = row[f"{chain_type}V"] if f"{chain_type}V" in row else None
                        features["v_gene"].append(v_gene)

        return pd.DataFrame(features)


    def _encode_repertoires(self, dataset: RepertoireDataset, params: EncoderParams):
        # Rows = repertoires, Columns = regex matches (one chain per column)
        encoded_repertoires = np.zeros((dataset.get_example_count(),
                                        self.feature_count),
                                       dtype=int)
        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        n_repertoires = dataset.get_example_count()

        for i, repertoire in enumerate(dataset.get_data()):
            print(f"{datetime.datetime.now()}: Encoding repertoire {i+1}/{n_repertoires}")
            encoded_repertoires[i] = self._match_repertoire_to_regexes(repertoire)

            if labels is not None:
                for label in params.label_config.get_labels_by_name():
                    labels[label].append(repertoire.metadata[label])

        return encoded_repertoires, labels

    def _match_repertoire_to_regexes(self, repertoire: Repertoire):
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences

        match_idx = 0


        for index, row in self.regex_df.iterrows():
            for chain_type in self.chains:
                regex = row[f"{chain_type}_regex"]

                if regex is not None:
                    v_gene = row[f"{chain_type}V"] if f"{chain_type}V" in row else None

                    for rep_seq in rep_seqs:
                        if rep_seq.metadata.chain.value == chain_type:
                            if self._matches(rep_seq, regex, v_gene):
                                # todo if count_counts instead of count_clones
                                matches[match_idx] += 1 if not self.sum_counts else rep_seq.metadata.count
                    match_idx += 1

        return matches


    def _matches(self, receptor_sequence, regex, v_gene=None):
        if v_gene is not None and receptor_sequence.metadata.v_gene != v_gene:
            matches = False
        else:
            matches = bool(re.search(regex, receptor_sequence.amino_acid_sequence))

        return matches