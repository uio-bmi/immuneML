import numpy as np
import pandas as pd

from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder


class MatchedSequencesRepertoireEncoder(MatchedSequencesEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)
        encoded_repertoires, labels = self._encode_repertoires(dataset, params)

        feature_annotations = self._get_feature_info()

        encoded_dataset.add_encoded_data(EncodedData(
            examples=encoded_repertoires,
            labels=labels,
            feature_names=list(feature_annotations["sequence_id"]),
            feature_annotations=feature_annotations,
            example_ids=[repertoire.identifier for repertoire in dataset.get_data()],
            encoding=MatchedSequencesEncoder.__name__
        ))

        return encoded_dataset

    def _get_feature_info(self):
        """
        returns a pandas dataframe containing:
         - sequence id
         - chain
         - amino acid sequence
         - v gene
         - j gene
        """

        features = [[] for i in range(0, len(self.reference_sequences))]

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
                                        len(self.reference_sequences)),
                                       dtype=int)

        labels = {label: [] for label in params.label_config.get_labels_by_name()} if params.encode_labels else None

        for i, repertoire in enumerate(dataset.get_data()):
            encoded_repertories[i] = self._match_repertoire_to_reference(repertoire)

            for label in params.label_config.get_labels_by_name():
                labels[label].append(repertoire.metadata[label])

        return encoded_repertories, labels

    def _match_repertoire_to_reference(self, repertoire: Repertoire):
        matcher = SequenceMatcher()
        matches = np.zeros(len(self.reference_sequences), dtype=int)
        rep_seqs = repertoire.sequences

        for i, reference_seq in enumerate(self.reference_sequences):

            for repertoire_seq in rep_seqs:
                if matcher.matches_sequence(reference_seq, repertoire_seq, max_distance=self.max_edit_distance):
                    matches[i] += repertoire_seq.metadata.count

        return matches
