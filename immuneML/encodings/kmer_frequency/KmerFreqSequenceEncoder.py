from collections import Counter

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.EncoderHelper import EncoderHelper


class KmerFreqSequenceEncoder(KmerFrequencyEncoder):

    def _encode_locus(self, dataset):
        return len(set(dataset.data.locus.tolist())) > 1

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset: SequenceDataset, params: EncoderParams):
        encoded_sequences = []
        sequence_ids = []

        encode_locus = self._encode_locus(dataset)
        sequence_encoder = self._prepare_sequence_encoder()
        params.region_type = self.region_type
        for sequence in dataset.get_data(region_type=self.region_type):
            counts = self._encode_sequence(sequence, params, sequence_encoder, Counter(), encode_locus)
            encoded_sequences.append(counts)
            sequence_ids.append(sequence.sequence_id)

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)
                  if params.encode_labels else None)

        return encoded_sequences, sequence_ids, labels
