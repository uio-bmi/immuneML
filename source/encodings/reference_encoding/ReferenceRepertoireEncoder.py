import numpy as np

from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReferenceEncoder import MatchedReferenceEncoder


class ReferenceRepertoireEncoder(MatchedReferenceEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        matched_info = self._match_repertories(dataset)

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires, params=dataset.params,
                                            metadata_file=dataset.metadata_file)
        encoded_repertoires, labels = self._encode_repertoires(dataset, matched_info, params)

        feature_name = self.summary.name.lower()

        encoded_dataset.add_encoded_data(EncodedData(
            examples=encoded_repertoires,
            labels=labels,
            feature_names=[feature_name],
            example_ids=[repertoire.identifier for repertoire in dataset.get_data()],
            encoding=MatchedReferenceEncoder.__name__
        ))

        self.store(encoded_dataset, params)
        return encoded_dataset

    def _encode_repertoires(self, dataset, matched_info, params: EncoderParams):
        encoded_repertories = np.zeros((dataset.get_example_count(), 1), dtype=float)
        labels = {label: [] for label in params["label_configuration"].get_labels_by_name()}

        for index, repertoire in enumerate(dataset.get_data()):
            assert repertoire.identifier == matched_info["repertoires"][index]["repertoire"], \
                "MatchedReferenceEncoder: error in SequenceMatcher ordering of repertoires."
            encoded_repertories[index] = matched_info["repertoires"][index][self.summary.name.lower()]
            for label_index, label in enumerate(params["label_configuration"].get_labels_by_name()):
                labels[label].append(repertoire.metadata[label])

        return np.reshape(encoded_repertories, newshape=(-1, 1)), labels

    def _match_repertories(self, dataset: RepertoireDataset):
        matcher = SequenceMatcher()
        matched_info = matcher.match(dataset=dataset,
                                     reference_sequences=self.reference_sequences,
                                     max_distance=self.max_edit_distance,
                                     summary_type=self.summary)
        return matched_info
