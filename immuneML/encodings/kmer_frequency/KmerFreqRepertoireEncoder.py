from multiprocessing.pool import Pool

import dill
import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.BNPSequenceEncodingStrategies import (
    dispatch_encoding, get_v_genes, kmer_weights, seq_field, V_GENE_ENCODING_TYPES,
)
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqRepertoireEncoder(KmerFrequencyEncoder):

    def _encode_locus(self, dataset):
        loci = [set(rep.data.locus.tolist()) for rep in dataset.repertoires]
        return len(set.union(*loci)) > 1

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data
        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):
        encode_locus = self._encode_locus(dataset)
        arguments = [(dill.dumps(rep), params, encode_locus) for rep in dataset.repertoires]

        with Pool(params.pool_size) as pool:
            repertoires = pool.starmap(self.get_encoded_repertoire, arguments)

        all_flat_kmers, all_weights, rep_ids_list, labels_list = zip(*repertoires)

        n_reps = len(all_flat_kmers)
        rep_kmer_counts = np.array([len(fk) for fk in all_flat_kmers], dtype=np.intp)
        flat_kmers = np.concatenate(all_flat_kmers) if any(len(fk) for fk in all_flat_kmers) else np.empty(0, dtype=str)
        row_ids = np.repeat(np.arange(n_reps), rep_kmer_counts)

        has_weights = any(w is not None for w in all_weights)
        if has_weights:
            weights_parts = [w if w is not None else np.ones(len(fk), dtype=float)
                             for fk, w in zip(all_flat_kmers, all_weights)]
            weights = np.concatenate(weights_parts) if any(len(p) for p in weights_parts) else None
        else:
            weights = None

        encoded_labels = {k: [d[k] for d in labels_list] for k in labels_list[0]} if params.encode_labels else None
        return flat_kmers, row_ids, weights, list(rep_ids_list), encoded_labels

    def get_encoded_repertoire(self, repertoire, params: EncoderParams, encode_locus: bool):
        rep = dill.loads(repertoire) if not isinstance(repertoire, Repertoire) else repertoire
        params.model = vars(self)
        return CacheHandler.memo_by_params(
            (("encoding_model", params.model), ("type", "kmer_encoding"),
             ("labels", params.label_config.get_labels_by_name()),
             ("repertoire_id", rep.identifier)),
            lambda: self._encode_single_repertoire(rep, params, encode_locus),
            CacheObjectType.ENCODING_STEP)

    def _encode_single_repertoire(self, rep: Repertoire, params: EncoderParams, encode_locus: bool):
        data = rep.data
        seq_array = getattr(data, seq_field(self.region_type, self.sequence_type))
        locus_labels = data.locus.tolist() if encode_locus else None
        v_genes = get_v_genes(data) if self.sequence_encoding in V_GENE_ENCODING_TYPES else None

        flat_kmers, row_ids = dispatch_encoding(
            seq_array, self.sequence_encoding,
            self.k, self.k_left, self.k_right, self.min_gap, self.max_gap,
            self.region_type, v_genes, locus_labels,
        )

        labels = None
        if params.encode_labels:
            labels = {name: rep.metadata[name] for name in params.label_config.get_labels_by_name()}

        return flat_kmers, kmer_weights(data, self.reads, row_ids), rep.identifier, labels