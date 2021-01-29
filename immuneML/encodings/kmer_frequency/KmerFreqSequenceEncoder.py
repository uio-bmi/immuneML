from collections import Counter

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqSequenceEncoder(KmerFrequencyEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = SequenceDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          labels=dataset.labels)

        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):

        encoded_sequences = []
        sequence_ids = []
        label_config = params.label_config
        labels = {label: [] for label in label_config.get_labels_by_name()} if params.encode_labels else None

        sequence_encoder = self._prepare_sequence_encoder()
        feature_names = sequence_encoder.get_feature_names(params)
        for sequence in dataset.get_data(params.pool_size):
            counts = self._encode_sequence(sequence, params, sequence_encoder, Counter())
            encoded_sequences.append(counts)
            sequence_ids.append(sequence.identifier)

            if params.encode_labels:
                for label_name in label_config.get_labels_by_name():
                    label = sequence.metadata.custom_params[label_name]
                    labels[label_name].append(label)

        return encoded_sequences, sequence_ids, labels, feature_names
