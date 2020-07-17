import hashlib
import math
from multiprocessing.pool import Pool

import numpy as np

from source.caching.CacheHandler import CacheHandler
from source.caching.CacheObjectType import CacheObjectType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.EncoderParams import EncoderParams
from source.encodings.onehot.OneHotEncoder import OneHotEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings


class OneHotRepertoireEncoder(OneHotEncoder):
    """
    One-hot encoded repertoire data is represented in a matrix with dimensions:
        [repertoires, sequences, sequence_lengths, one_hot_characters]

    when use_positional_info is true, the last 3 indices in one_hot_characters represents the positional information:
        - start position (high when close to start)
        - middle position (high in the middle of the sequence)
        - end position (high when close to end)
    """

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires,
                                            encoded_data=encoded_data,
                                            params=dataset.params,
                                            metadata_file=dataset.metadata_file)

        self.store(encoded_dataset, params)

        return encoded_dataset

    def _set_max_dims(self, dataset):
        max_rep_len = 0
        max_seq_len = 0

        for repertoire in dataset.repertoires:
            sequences = repertoire.get_attribute(EnvironmentSettings.get_sequence_type().value)
            max_rep_len = max(len(sequences), max_rep_len)
            max_seq_len = max(max([len(seq) for seq in sequences]), max_seq_len)

        self.max_rep_len = max_rep_len
        self.max_seq_len = max_seq_len

    def _encode_data(self, dataset, params: EncoderParams):
        self._set_max_dims(dataset)

        arguments = [(repertoire, params) for repertoire in dataset.repertoires]

        with Pool(params["batch_size"]) as pool:
            chunksize = math.floor(dataset.get_example_count() / params["batch_size"]) + 1
            repertoires = pool.starmap(self._get_encoded_repertoire, arguments, chunksize=chunksize)

        encoded_repertoires, repertoire_names, labels = zip(*repertoires)

        encoded_dataset = np.stack(encoded_repertoires, axis=0)

        labels = {k: [dic[k] for dic in labels] for k in labels[0]}

        encoded_data = EncodedData(examples=encoded_dataset,
                                   example_ids=repertoire_names,
                                   labels=labels,
                                   encoding=OneHotEncoder.__name__)

        return encoded_data

    def _get_encoded_repertoire(self, repertoire, params: EncoderParams):
        params["model"] = vars(self)

        return CacheHandler.memo_by_params((("encoding_model", params["model"]),
                                            ("labels", params["label_configuration"].get_labels_by_name()),
                                            ("repertoire_id", repertoire.identifier),
                                            ("repertoire_data", hashlib.sha256(np.ascontiguousarray(repertoire.get_sequence_aas())).hexdigest())),
                                           lambda: self._encode_repertoire(repertoire, params), CacheObjectType.ENCODING)

    def _encode_repertoire(self, repertoire, params: EncoderParams):
        sequences = repertoire.get_attribute(EnvironmentSettings.get_sequence_type().value)

        onehot_encoded = self._encode_sequence_list(sequences, pad_n_sequences=self.max_rep_len, pad_sequence_len=self.max_seq_len)
        example_id = repertoire.identifier
        labels = self._get_repertoire_labels(repertoire, params)

        return onehot_encoded, example_id, labels

    def _get_repertoire_labels(self, repertoire, params: EncoderParams):
        label_config = params["label_configuration"]
        labels = dict()

        for label_name in label_config.get_labels_by_name():
            labels[label_name] = repertoire.metadata[label_name]

        return labels
