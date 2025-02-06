from collections import Counter
from multiprocessing.pool import Pool

import dill

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqRepertoireEncoder(KmerFrequencyEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):

        arguments = [(dill.dumps(repertoire), params) for repertoire in dataset.repertoires]

        with Pool(params.pool_size) as pool:
            repertoires = pool.starmap(self.get_encoded_repertoire, arguments)

        encoded_repertoire_list, repertoire_names, labels, feature_annotation_names = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]} if params.encode_labels else None

        feature_annotation_names = feature_annotation_names[0]

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels, feature_annotation_names

    def get_encoded_repertoire(self, repertoire, params: EncoderParams):

        if not isinstance(repertoire, Repertoire):
            rep = dill.loads(repertoire)
        else:
            rep = repertoire

        params.model = vars(self)

        return CacheHandler.memo_by_params((("encoding_model", params.model), ("type", "kmer_encoding"),
                                            ("labels", params.label_config.get_labels_by_name()),
                                            ("repertoire_id", rep.identifier)),
                                           lambda: self.encode_repertoire(rep, params), CacheObjectType.ENCODING_STEP)

    def encode_repertoire(self, repertoire, params: EncoderParams):
        counts = Counter()
        sequence_encoder = self._prepare_sequence_encoder()
        feature_names = sequence_encoder.get_feature_names(params)
        params.region_type = self.region_type
        for sequence in repertoire.sequences(params.region_type):
            counts = self._encode_sequence(sequence, params, sequence_encoder, counts)

        label_config = params.label_config
        labels = dict() if params.encode_labels else None

        if labels is not None:
            for label_name in label_config.get_labels_by_name():
                label = repertoire.metadata[label_name]
                labels[label_name] = label

        # TODO: refactor this not to return 4 values but e.g. a dict or split into different functions?
        return counts, repertoire.identifier, labels, feature_names
