from collections import Counter

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqSequenceEncoder(KmerFrequencyEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset: SequenceDataset, params: EncoderParams):
        sequence_encoder = self._prepare_sequence_encoder()
        
        # Try using optimized k-mer computation
        counts = self._process_sequences(dataset.data, params)
        
        if counts is not None:
            # Use optimized results
            encoded_sequences = counts
            sequence_ids = dataset.data.sequence_id
            
            # Get labels if needed
            label_config = params.label_config
            labels = {label: [] for label in label_config.get_labels_by_name()} if params.encode_labels else None

            if params.encode_labels:
                for label_name in label_config.get_labels_by_name():
                    labels[label_name] = getattr(dataset.data, label_name)
        else:
            # Use traditional encoding
            encoded_sequences = []
            sequence_ids = []
            label_config = params.label_config
            labels = {label: [] for label in label_config.get_labels_by_name()} if params.encode_labels else None

            for sequence in dataset.get_data(region_type=self.region_type):
                counts = self._encode_sequence(sequence, params, sequence_encoder, Counter())
                encoded_sequences.append(counts)
                sequence_ids.append(sequence.sequence_id)

                if params.encode_labels:
                    for label_name in label_config.get_labels_by_name():
                        label = sequence.metadata[label_name]
                        labels[label_name].append(label)

        return encoded_sequences, sequence_ids, labels, sequence_encoder.get_feature_names(params)
