import numpy as np

from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.BNPSequenceEncodingStrategies import (
    dispatch_encoding, get_v_genes, kmer_weights, seq_field, V_GENE_ENCODING_TYPES,
)
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.EncoderHelper import EncoderHelper


def _encode_chain(data, chain_mask: np.ndarray, locus_name: str,
                   encoder: KmerFrequencyEncoder) -> tuple:
    seq_array = getattr(data, seq_field(encoder.region_type, encoder.sequence_type))[chain_mask]
    locus_labels = [locus_name] * int(chain_mask.sum())
    v_genes = get_v_genes(data, chain_mask) if encoder.sequence_encoding in V_GENE_ENCODING_TYPES else None

    flat_kmers, row_ids = dispatch_encoding(
        seq_array, encoder.sequence_encoding,
        encoder.k, encoder.k_left, encoder.k_right, encoder.min_gap, encoder.max_gap,
        encoder.region_type, v_genes, locus_labels,
    )

    return flat_kmers, row_ids, kmer_weights(data, encoder.reads, row_ids, chain_mask)


class KmerFreqReceptorEncoder(KmerFrequencyEncoder):

    def _encode_locus(self, dataset):
        return True

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data
        return encoded_dataset

    def _encode_examples(self, dataset: ReceptorDataset, params: EncoderParams):
        data = dataset.data
        receptor_ids, loci, mask1, mask2 = EncoderHelper.get_receptor_chain_masks(dataset)

        all_flat, all_row_ids, all_weights = [], [], []
        for mask, locus in ((mask1, loci[0]), (mask2, loci[1])):
            flat_kmers, row_ids, weights = _encode_chain(data, mask, locus, self)
            if len(flat_kmers) > 0:
                all_flat.append(flat_kmers)
                all_row_ids.append(row_ids)
                if weights is not None:
                    all_weights.append(weights)

        flat_kmers = np.concatenate(all_flat) if all_flat else np.empty(0, dtype=str)
        row_ids = np.concatenate(all_row_ids) if all_row_ids else np.empty(0, dtype=np.intp)
        weights = np.concatenate(all_weights) if all_weights else None

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)
                  if params.encode_labels else None)

        return flat_kmers, row_ids, weights, receptor_ids, labels