from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.BNPSequenceEncodingStrategies import (
    dispatch_encoding, get_v_genes, kmer_weights, seq_field, V_GENE_ENCODING_TYPES,
)
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
        data = dataset.data
        seq_array = getattr(data, seq_field(self.region_type, self.sequence_type))

        encode_locus = self._encode_locus(dataset)
        locus_labels = data.locus.tolist() if encode_locus else None
        v_genes = get_v_genes(data) if self.sequence_encoding in V_GENE_ENCODING_TYPES else None

        flat_kmers, row_ids = dispatch_encoding(
            seq_array, self.sequence_encoding,
            self.k, self.k_left, self.k_right, self.min_gap, self.max_gap,
            self.region_type, v_genes, locus_labels,
        )

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)
                  if params.encode_labels else None)

        return flat_kmers, row_ids, kmer_weights(data, self.reads, row_ids), data.sequence_id.tolist(), labels