import hashlib
from multiprocessing.pool import Pool

import math
from typing import Union

import dill
import numpy as np

from immuneML.analysis.entropy_calculations.EntropyCalculator import EntropyCalculator
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
from immuneML.util.Logger import log


class EvennessProfileRepertoireEncoder(EvennessProfileEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):

        arguments = [(repertoire.identifier, dill.dumps(repertoire), params) for repertoire in dataset.repertoires]

        with Pool(params.pool_size) as pool:
            chunksize = math.floor(dataset.get_example_count()/params.pool_size) + 1
            repertoires = pool.starmap(self.get_encoded_repertoire, arguments, chunksize=chunksize)

        encoded_repertoire_list, repertoire_names, labels = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]} if params.encode_labels else None

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels

    def get_encoded_repertoire(self, repertoire_id: str, repertoire: Union[bytes, Repertoire], params: EncoderParams):

        params.model = vars(self)
        serialized_repertoire = dill.dumps(repertoire) if isinstance(repertoire, Repertoire) else repertoire

        return CacheHandler.memo_by_params((("encoding_model", params.model),
                                            ("labels", params.label_config.get_labels_by_name()),
                                            ("repertoire_id", repertoire_id)),
                                           lambda: self.encode_repertoire(serialized_repertoire, params),
                                           CacheObjectType.ENCODING_STEP)

    def encode_repertoire(self, repertoire, params: EncoderParams):
        if isinstance(repertoire, bytes):
            repertoire = dill.loads(repertoire)

        alphas = np.linspace(start=params.model["min_alpha"], stop=params.model["max_alpha"], num=params.model["dimension"])

        data = repertoire.data
        counts = data.duplicate_count[np.array(data.vj_in_frame == 'T').flatten()]

        freqs = counts[np.nonzero(counts)]

        evenness_profile = np.array([np.exp(EntropyCalculator.renyi_entropy(freqs, alpha))/len(freqs) for alpha in alphas])

        if params.encode_labels:
            label_config = params.label_config
            labels = dict()
            for label_name in label_config.get_labels_by_name():
                label = repertoire.metadata[label_name]
                labels[label_name] = label
        else:
            labels = None

        return evenness_profile, repertoire.identifier, labels
