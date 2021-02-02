import abc
import math

import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from immuneML.IO.dataset_export.PickleExporter import PickleExporter
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class OneHotEncoder(DatasetEncoder):
    """
    One-hot encoding for repertoires, sequences or receptors. In one-hot encoding, each alphabet character
    (amino acid or nucleotide) is replaced by a sparse vector with one 1 and the rest zeroes. The position of the
    1 represents the alphabet character.


    Arguments:

        use_positional_info (bool): whether to include features representing the positional information.
        If True, three additional feature vectors will be added, representing the sequence start, sequence middle
        and sequence end. The values in these features are scaled between 0 and 1. A graphical representation of
        the values of these vectors is given below.

        .. code-block:: console

              Value of sequence start:         Value of sequence middle:        Value of sequence end:

            1 \                              1    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\         1                          /
               \                                 /                   \                                  /
                \                               /                     \                                /
            0    \_____________________      0 /                       \      0  _____________________/
              <----sequence length---->        <----sequence length---->         <----sequence length---->


        distance_to_seq_middle (int): only applies when use_positional_info is True. This is the distance from the edge
        of the CDR3 sequence (IMGT positions 105 and 117) to the portion of the sequence that is considered 'middle'.
        For example: if distance_to_seq_middle is 6 (default), all IMGT positions in the interval [111, 112)
        receive positional value 1.
        When using nucleotide sequences: note that the distance is measured in (amino acid) IMGT positions.
        If the complete sequence length is smaller than 2 * distance_to_seq_middle, the maximum value of the
        'start' and 'end' vectors will not reach 0, and the maximum value of the 'middle' vector will not reach 1.
        A graphical representation of teh positional vectors with a too short sequence is given below:


        .. code-block:: console

            Value of sequence start         Value of sequence middle        Value of sequence end:
            with very short sequence:       with very short sequence:       with very short sequence:

                 1 \                               1                                 1    /
                    \                                                                    /
                     \                                /\                                /
                 0                                 0 /  \                            0
                   <->                               <-->                               <->

        flatten (bool): whether to flatten the final onehot matrix to a 2-dimensional matrix [examples, other_dims_combined]
        This must be set to True when using onehot encoding in combination with scikit-learn ML methods (inheriting :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`),
        such as :ref:`LogisticRegression`, :ref:`SVM`, :ref:`RandomForestClassifier` and :ref:`KNN`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        one_hot_vanilla:
            OneHot:
                use_positional_info: False
                flatten: False

        one_hot_positional:
            OneHot:
                use_positional_info: True
                distance_to_seq_middle: 3
                flatten: False

    """

    dataset_mapping = {
        "RepertoireDataset": "OneHotRepertoireEncoder",
        "SequenceDataset": "OneHotSequenceEncoder",
        "ReceptorDataset": "OneHotReceptorEncoder"
    }

    ALPHABET = EnvironmentSettings.get_sequence_alphabet()

    def __init__(self, use_positional_info: bool, distance_to_seq_middle: int, flatten: bool, name: str = None):
        self.use_positional_info = use_positional_info
        self.distance_to_seq_middle = distance_to_seq_middle
        self.flatten = flatten

        if distance_to_seq_middle:
            self.pos_increasing = [1 / self.distance_to_seq_middle * i for i in range(self.distance_to_seq_middle)]
            self.pos_decreasing = self.pos_increasing[::-1]
        else:
            self.pos_decreasing = None

        self.name = name

        if EnvironmentSettings.get_sequence_type() == SequenceType.NUCLEOTIDE: # todo check this / explain in docs
            self.distance_to_seq_middle = self.distance_to_seq_middle * 3

        self.onehot_dimensions = self.ALPHABET + ["start", "mid", "end"] if self.use_positional_info else self.ALPHABET # todo test this

    @staticmethod
    def _prepare_parameters(use_positional_info, distance_to_seq_middle, flatten, name: str = None):

        location = OneHotEncoder.__name__

        ParameterValidator.assert_type_and_value(use_positional_info, bool, location, "use_positional_info")
        if use_positional_info:
            ParameterValidator.assert_type_and_value(distance_to_seq_middle, int, location, "distance_to_seq_middle", min_inclusive=1)
        else:
            distance_to_seq_middle = None

        ParameterValidator.assert_type_and_value(flatten, bool, location, "flatten")

        return {"use_positional_info": use_positional_info,
                "distance_to_seq_middle": distance_to_seq_middle,
                "flatten": flatten,
                "name": name}

    @staticmethod
    def build_object(dataset=None, **params):

        try:
            prepared_params = OneHotEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(OneHotEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "onehot/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(OneHotEncoder.__name__, dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):
        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file if hasattr(dataset, "metadata_file") else None),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", OneHotEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", step),
                ("encoding_params", tuple(vars(self).items())))

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params.result_path)

    def _encode_sequence_list(self, sequences, pad_n_sequences, pad_sequence_len):
        char_array = np.array(sequences, dtype=str)
        char_array = char_array.view('U1').reshape((char_array.size, -1))

        n_sequences, sequence_len = char_array.shape

        sklearn_enc = SklearnOneHotEncoder(categories=[OneHotEncoder.ALPHABET for i in range(sequence_len)], handle_unknown='ignore')
        encoded_data = sklearn_enc.fit_transform(char_array).toarray()

        encoded_data = np.pad(encoded_data, pad_width=((0, pad_n_sequences - n_sequences), (0, 0)))
        encoded_data = encoded_data.reshape((pad_n_sequences, sequence_len, len(OneHotEncoder.ALPHABET)))
        positional_dims = int(self.use_positional_info) * 3
        encoded_data = np.pad(encoded_data, pad_width=((0, 0), (0, pad_sequence_len - sequence_len), (0, positional_dims)))

        if self.use_positional_info:
            pos_info = [self._get_imgt_position_weights(len(sequence), pad_length=pad_sequence_len).T for sequence in sequences]
            pos_info = np.stack(pos_info)
            pos_info = np.pad(pos_info, pad_width=((0, pad_n_sequences - n_sequences), (0, 0), (0, 0)))

            encoded_data[:, :, len(OneHotEncoder.ALPHABET):] = pos_info

        return encoded_data

    def _get_imgt_position_weights(self, seq_length, pad_length=None):
        start_weights = self._get_imgt_start_weights(seq_length)
        mid_weights = self._get_imgt_mid_weights(seq_length)
        end_weights = start_weights[::-1]

        weights = np.array([start_weights, mid_weights, end_weights])

        if pad_length is not None:
            weights = np.pad(weights, pad_width=((0, 0), (0, pad_length - seq_length)))

        return weights

    def _get_imgt_mid_weights(self, seq_length):
        mid_len = seq_length - (self.distance_to_seq_middle * 2)

        if mid_len >= 0:
            mid_weights = self.pos_increasing + [1] * mid_len + self.pos_decreasing
        else:
            left_idx = math.ceil(seq_length / 2)
            right_idx = math.floor(seq_length / 2)

            mid_weights = self.pos_increasing[:left_idx] + self.pos_decreasing[-right_idx:]

        return mid_weights

    def _get_imgt_start_weights(self, seq_length):
        diff = (seq_length - self.distance_to_seq_middle) - 1
        if diff >= 0:
            start_weights = [1] + self.pos_decreasing + [0] * diff
        else:
            start_weights = [1] + self.pos_decreasing[:diff]

        return start_weights
