import numpy as np
import pandas as pd

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.ReadsType import ReadsType
from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder


class MatchedReceptorsRepertoireEncoder(MatchedReceptorsEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)

        feature_annotations = None if self.sum_matches else self._get_feature_info()

        if self.sum_matches:
            chains = self.reference_receptors[0].get_chains()
            feature_names = [f"sum_of_{self.reads.value}_reads_{chains[0]}", f"sum_of_{self.reads.value}_reads_{chains[1]}"]
        else:
            feature_names = [f"{row['receptor_id']}.{row['chain']}" for index, row in feature_annotations.iterrows()]

        encoded_repertoires, labels, example_ids = self._encode_repertoires(dataset, params)
        encoded_repertoires = self._normalize(dataset, encoded_repertoires) if self.normalize else encoded_repertoires

        encoded_dataset.add_encoded_data(EncodedData(
            # examples contains a np.ndarray with counts
            examples=encoded_repertoires,
            # example_ids contains a list of repertoire identifiers
            example_ids=example_ids,
            # feature_names contains a list of reference receptor identifiers
            feature_names=feature_names,
            # feature_annotations contains a PD dataframe with sequence and VDJ gene usage per reference receptor
            feature_annotations=feature_annotations,
            labels=labels,
            encoding=MatchedReceptorsEncoder.__name__
        ))

        return encoded_dataset

    def _normalize(self, dataset, encoded_repertoires):
        if self.reads == ReadsType.UNIQUE:
            repertoire_totals = np.asarray([[repertoire.get_element_count() for repertoire in dataset.get_data()]]).T
        else:
            repertoire_totals = np.asarray([[sum(repertoire.get_counts()) for repertoire in dataset.get_data()]]).T

        return encoded_repertoires / repertoire_totals


    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - receptor id
         - receptor chain
         - amino acid sequence
         - v gene
         - j gene
        """

        features = [[] for i in range(0, self.feature_count)]

        for i, receptor in enumerate(self.reference_receptors):
            id = receptor.identifier
            chain_names = receptor.get_chains()
            first_chain = receptor.get_chain(chain_names[0])
            second_chain = receptor.get_chain(chain_names[1])

            clonotype_id = receptor.metadata["clonotype_id"] if "clonotype_id" in receptor.metadata else None

            if first_chain.metadata.custom_params is not None:
                first_dual_chain_id = first_chain.metadata.custom_params["dual_chain_id"] if "dual_chain_id" in first_chain.metadata.custom_params else None

            if second_chain.metadata.custom_params is not None:
                second_dual_chain_id = second_chain.metadata.custom_params["dual_chain_id"] if "dual_chain_id" in second_chain.metadata.custom_params else None

            features[i * 2] = [id, clonotype_id, chain_names[0],
                               first_dual_chain_id,
                               first_chain.amino_acid_sequence,
                               first_chain.metadata.v_gene,
                               first_chain.metadata.j_gene]
            features[i * 2 + 1] = [id, clonotype_id, chain_names[1],
                                   second_dual_chain_id,
                                   second_chain.amino_acid_sequence,
                                   second_chain.metadata.v_gene,
                                   second_chain.metadata.j_gene]

        features = pd.DataFrame(features,
                                columns=["receptor_id", "clonotype_id", "chain", "dual_chain_id", "sequence", "v_gene", "j_gene"])

        features.dropna(axis="columns", how="all", inplace=True)

        return features

    def _encode_repertoires(self, dataset: RepertoireDataset, params: EncoderParams):
        # Rows = repertoires, Columns = reference chains (two per sequence receptor)
        encoded_repertories = np.zeros((dataset.get_example_count(),
                                        self.feature_count),
                                       dtype=int)
        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        for i, repertoire in enumerate(dataset.get_data()):
            encoded_repertories[i] = self._match_repertoire_to_receptors(repertoire)

            if labels is not None:
                for label_name in params.label_config.get_labels_by_name():
                    labels[label_name].append(repertoire.metadata[label_name])

        return encoded_repertories, labels, dataset.get_repertoire_ids()

    def _match_repertoire_to_receptors(self, repertoire: Repertoire):
        matcher = SequenceMatcher()
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences

        for i, ref_receptor in enumerate(self.reference_receptors):
            chain_names = ref_receptor.get_chains()
            first_chain = ref_receptor.get_chain(chain_names[0])
            second_chain = ref_receptor.get_chain(chain_names[1])

            for rep_seq in rep_seqs:
                matches_idx = 0 if self.sum_matches else i * 2
                match_count = 1 if self.reads == ReadsType.UNIQUE else rep_seq.metadata.count

                # Match with first chain: add to even columns in matches.
                # Match with second chain: add to odd columns
                if matcher.matches_sequence(first_chain, rep_seq, max_distance=self.max_edit_distances[chain_names[0]]):
                    matches[matches_idx] += match_count
                if matcher.matches_sequence(second_chain, rep_seq, max_distance=self.max_edit_distances[chain_names[1]]):
                    matches[matches_idx + 1] += match_count

        return matches

