import hashlib
import math
from multiprocessing.pool import Pool

import numpy as np

from immuneML.analysis.entropy_calculations.EntropyCalculator import EntropyCalculator
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
from immuneML.util.Logger import log


class EvennessProfileRepertoireEncoder(EvennessProfileEncoder):

    @log
    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires,
                                            encoded_data=encoded_data,
                                            labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)

        return encoded_dataset

    @log
    def _encode_examples(self, dataset, params: EncoderParams):

        arguments = [(repertoire, params) for repertoire in dataset.repertoires]

        with Pool(params.pool_size) as pool:
            chunksize = math.floor(dataset.get_example_count()/params.pool_size) + 1
            repertoires = pool.starmap(self.get_encoded_repertoire, arguments, chunksize=chunksize)

        encoded_repertoire_list, repertoire_names, labels = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]} if params.encode_labels else None

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels

    def get_encoded_repertoire(self, repertoire, params: EncoderParams):

        params.model = vars(self)

        return CacheHandler.memo_by_params((("encoding_model", params.model),
                                            ("labels", params.label_config.get_labels_by_name()),
                                            ("repertoire_id", repertoire.identifier),
                                            ("repertoire_data",  hashlib.sha256(np.ascontiguousarray(repertoire.get_sequence_aas())).hexdigest())),
                                           lambda: self.encode_repertoire(repertoire, params), CacheObjectType.ENCODING_STEP)

    def encode_repertoire(self, repertoire, params: EncoderParams):

        alphas = np.linspace(start=params.model["min_alpha"], stop=params.model["max_alpha"], num=params.model["dimension"])

        counts = [sequence.metadata.count for sequence in repertoire.sequences if sequence.metadata.frame_type == SequenceFrameType.IN]
        freqs = np.array(counts)
        freqs = freqs[np.nonzero(freqs)]

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
