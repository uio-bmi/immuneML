import hashlib
import math
from collections import Counter
from multiprocessing.pool import Pool

import numpy as np

from source.caching.CacheHandler import CacheHandler
from source.caching.CacheObjectType import CacheObjectType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.logging.Logger import log


class KmerFreqRepertoireEncoder(KmerFrequencyEncoder):

    @log
    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires,
                                            encoded_data=encoded_data,
                                            params=dataset.params,
                                            metadata_file=dataset.metadata_file)

        self.store(encoded_dataset, params)

        return encoded_dataset

    @log
    def _encode_examples(self, dataset, params: EncoderParams):

        arguments = [(repertoire, params) for repertoire in dataset.repertoires]

        with Pool(params["batch_size"]) as pool:
            chunksize = math.floor(dataset.get_example_count()/params["batch_size"]) + 1
            repertoires = pool.starmap(self._get_encoded_repertoire, arguments, chunksize=chunksize)

        encoded_repertoire_list, repertoire_names, labels, feature_annotation_names = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]}

        feature_annotation_names = feature_annotation_names[0]

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels, feature_annotation_names

    def _get_encoded_repertoire(self, repertoire, params: EncoderParams):
        params["model"] = vars(self)

        return CacheHandler.memo_by_params((("encoding_model", params["model"]),
                                            ("labels", params["label_configuration"].get_labels_by_name()),
                                            ("repertoire_id", repertoire.identifier),
                                            ("repertoire_data",  hashlib.sha256(np.ascontiguousarray(repertoire.get_sequence_aas())).hexdigest())),
                                           lambda: self._encode_repertoire(repertoire, params), CacheObjectType.ENCODING_STEP)

    def _encode_repertoire(self, repertoire, params: EncoderParams):
        counts = Counter()
        sequence_encoder = self._prepare_sequence_encoder(params)
        feature_names = sequence_encoder.get_feature_names(params)
        for sequence in repertoire.sequences:
            counts = self._encode_sequence(sequence, params, sequence_encoder, counts)

        label_config = params["label_configuration"]
        labels = dict()

        for label_name in label_config.get_labels_by_name():
            label = repertoire.metadata[label_name]
            labels[label_name] = label

        # TODO: refactor this not to return 4 values but e.g. a dict or split into different functions?
        return counts, repertoire.identifier, labels, feature_names
