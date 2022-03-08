import numpy as np
import pandas as pd

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.ReadsType import ReadsType
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder


class MatchedSequencesRepertoireEncoder(MatchedSequencesEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)

        encoded_repertoires, labels = self._encode_repertoires(dataset, params)

        encoded_repertoires = self._normalize(dataset, encoded_repertoires) if self.normalize else encoded_repertoires

        feature_annotations = None if self.sum_matches else self._get_feature_info()
        feature_names = [f"sum_of_{self.reads.value}_reads"] if self.sum_matches else list(feature_annotations["sequence_id"])

        encoded_dataset.add_encoded_data(EncodedData(
            examples=encoded_repertoires,
            labels=labels,
            feature_names=feature_names,
            feature_annotations=feature_annotations,
            example_ids=[repertoire.identifier for repertoire in dataset.get_data()],
            encoding=MatchedSequencesEncoder.__name__
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
         - sequence id
         - chain
         - amino acid sequence
         - v gene
         - j gene
        """

        features = [[] for i in range(0, self.feature_count)]

        for i, sequence in enumerate(self.reference_sequences):
            features[i] = [sequence.identifier,
                           sequence.get_attribute("chain").name.lower(),
                           sequence.get_sequence(),
                           sequence.get_attribute("v_gene"),
                           sequence.get_attribute("j_gene")]

        features = pd.DataFrame(features,
                                columns=["sequence_id", "chain", "sequence", "v_gene", "j_gene"])

        return features

    def _encode_repertoires(self, dataset: RepertoireDataset, params):
        # Rows = repertoires, Columns = reference sequences
        encoded_repertories = np.zeros((dataset.get_example_count(),
                                        self.feature_count),
                                       dtype=int)

        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        for i, repertoire in enumerate(dataset.get_data()):
            encoded_repertories[i] = self._match_repertoire_to_reference(repertoire)

            for label_name in params.label_config.get_labels_by_name():
                labels[label_name].append(repertoire.metadata[label_name])

        return encoded_repertories, labels

    def _match_repertoire_to_reference(self, repertoire: Repertoire):
        matcher = SequenceMatcher()
        matches = np.zeros(self.feature_count, dtype=int)
        rep_seqs = repertoire.sequences

        for i, reference_seq in enumerate(self.reference_sequences):

            for repertoire_seq in rep_seqs:
                if matcher.matches_sequence(reference_seq, repertoire_seq, max_distance=self.max_edit_distance):
                    matches_idx = 0 if self.sum_matches else i
                    match_count = 1 if self.reads == ReadsType.UNIQUE else repertoire_seq.metadata.count
                    matches[matches_idx] += match_count

        return matches
