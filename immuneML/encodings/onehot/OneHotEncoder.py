import abc
import math

import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model import bnp_util
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class OneHotEncoder(DatasetEncoder):
    """
    One-hot encoding for repertoires, sequences or receptors. In one-hot encoding, each alphabet character
    (amino acid or nucleotide) is replaced by a sparse vector with one 1 and the rest zeroes. The position of the
    1 represents the alphabet character.

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets

    - RepertoireDatasets


    **Specification arguments:**

    - use_positional_info (bool): whether to include features representing the positional information.
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


    - distance_to_seq_middle (int): only applies when use_positional_info is True. This is the distance from the edge
      of the CDR3 sequence (IMGT positions 105 and 117) to the portion of the sequence that is considered 'middle'.
      For example: if distance_to_seq_middle is 6 (default), all IMGT positions in the interval [111, 112)
      receive positional value 1.
      When using nucleotide sequences: note that the distance is measured in (amino acid) IMGT positions.
      If the complete sequence length is smaller than 2 * distance_to_seq_middle, the maximum value of the
      'start' and 'end' vectors will not reach 0, and the maximum value of the 'middle' vector will not reach 1.
      A graphical representation of the positional vectors with a too short sequence is given below:


    .. code-block:: console

        Value of sequence start         Value of sequence middle        Value of sequence end:
        with very short sequence:       with very short sequence:       with very short sequence:

             1 \                               1                                 1    /
                \                                                                    /
                 \                                /\                                /
             0                                 0 /  \                            0
               <->                               <-->                               <->

    - flatten (bool): whether to flatten the final onehot matrix to a 2-dimensional matrix [examples, other_dims_combined]
      This must be set to True when using onehot encoding in combination with scikit-learn ML methods (inheriting :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`),
      such as :ref:`LogisticRegression`, :ref:`SVM`, :ref:`SVC`, :ref:`RandomForestClassifier` and :ref:`KNN`.

    - sequence_type: whether to use nucleotide or amino acid sequence for encoding. Valid values are 'nucleotide' and 'amino_acid'.

    - region_type: which part of the sequence to encode; e.g., imgt_cdr3, imgt_junction


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                one_hot_vanilla:
                    OneHot:
                        use_positional_info: False
                        flatten: False
                        sequence_type: amino_acid
                        region_type: imgt_cdr3

                one_hot_positional:
                    OneHot:
                        use_positional_info: True
                        distance_to_seq_middle: 3
                        flatten: False
                        sequence_type: nucleotide

    """

    dataset_mapping = {
        "RepertoireDataset": "OneHotRepertoireEncoder",
        "SequenceDataset": "OneHotSequenceEncoder",
        "ReceptorDataset": "OneHotReceptorEncoder"
    }

    def __init__(self, use_positional_info: bool, distance_to_seq_middle: int, flatten: bool, name: str = None,
                 sequence_type: SequenceType = None, region_type: RegionType = None):
        super().__init__(name=name)
        self.use_positional_info = use_positional_info
        self.distance_to_seq_middle = distance_to_seq_middle
        self.flatten = flatten
        self.sequence_type = sequence_type
        self.region_type = region_type
        self.alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)

        if distance_to_seq_middle:
            self.pos_increasing = [1 / self.distance_to_seq_middle * i for i in range(self.distance_to_seq_middle)]
            self.pos_decreasing = self.pos_increasing[::-1]
        else:
            self.pos_decreasing = None

        if self.sequence_type == SequenceType.NUCLEOTIDE and self.distance_to_seq_middle is not None:  # todo check this / explain in docs
            self.distance_to_seq_middle = self.distance_to_seq_middle * 3

        if self.use_positional_info:
            self.onehot_dimensions = self.alphabet + ["start", "mid", "end"]
        else:
            self.onehot_dimensions = self.alphabet

    @staticmethod
    def _prepare_parameters(use_positional_info: bool, distance_to_seq_middle: int, flatten: bool, sequence_type: str,
                            name: str = None, region_type: str = None):

        location = OneHotEncoder.__name__

        ParameterValidator.assert_type_and_value(use_positional_info, bool, location, "use_positional_info")
        if use_positional_info:
            ParameterValidator.assert_type_and_value(distance_to_seq_middle, int, location, "distance_to_seq_middle",
                                                     min_inclusive=1)
        else:
            distance_to_seq_middle = None

        ParameterValidator.assert_type_and_value(flatten, bool, location, "flatten")
        ParameterValidator.assert_type_and_value(sequence_type, str, location, 'sequence_type')
        ParameterValidator.assert_in_valid_list(sequence_type.upper(), [item.name for item in SequenceType], location,
                                                'sequence_type')
        ParameterValidator.assert_region_type({'region_type': region_type}, location)

        return {"use_positional_info": use_positional_info, "region_type": RegionType[region_type.upper()],
                "distance_to_seq_middle": distance_to_seq_middle,
                "flatten": flatten, "sequence_type": SequenceType[sequence_type.upper()],
                "name": name}

    @staticmethod
    def build_object(dataset=None, **params):
        EncoderHelper.check_dataset_type_available_in_mapping(dataset, OneHotEncoder)

        prepared_params = OneHotEncoder._prepare_parameters(**params)
        encoder = ReflectionHandler.get_class_by_name(OneHotEncoder.dataset_mapping[dataset.__class__.__name__],
                                                      "onehot/")(**prepared_params)
        return encoder

    def encode(self, dataset, params: EncoderParams):
        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):
        return (("dataset_identifier", dataset.identifier),
                ("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", OneHotEncoder.__name__),
                ("encoding_params", tuple(vars(self).items())))

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    def _get_seq_field_name(self, params: EncoderParams) -> str:
        region_type = self.region_type if self.region_type else params.region_type
        sequence_type = self.sequence_type if self.sequence_type else params.sequence_type
        return bnp_util.get_sequence_field_name(region_type, sequence_type)

    def _encode_sequence_list(self, sequences: AIRRSequenceSet, params: EncoderParams, pad_n_sequences: int = None,
                              pad_sequence_len: int = None):
        # Get sequence field based on sequence_type
        sequence_field = self._get_seq_field_name(params)

        # Extract sequences from AIRRSequenceSet
        sequence_array = getattr(sequences, sequence_field)

        sequence_alphabet = "".join(EnvironmentSettings.get_sequence_alphabet(params.sequence_type))

        # noinspection PyTypeChecker
        encoded_sequences = np.array([
            np.pad(np.array((sequence_array[i][..., np.newaxis] == sequence_alphabet).tolist()),
                   [[0, pad_sequence_len - sequence_array.lengths[i]], [0, 0]], mode='constant',
                   constant_values=False) for i in range(len(sequence_array))
        ])

        if encoded_sequences.shape[0] != pad_n_sequences:
            encoded_sequences = np.concatenate([encoded_sequences,
                                                np.zeros(
                                                    (pad_n_sequences - encoded_sequences.shape[0], pad_sequence_len,
                                                     len(sequence_alphabet)))], axis=0)

        # Add positional encoding if needed
        if self.use_positional_info:
            pos_info = ([self._get_imgt_position_weights(seq_len,
                                                         pad_length=pad_sequence_len).T
                         for seq_len in sequence_array.lengths]
                        + [[[0, 0, 0] for _ in range(pad_sequence_len)] for s in range(pad_n_sequences - len(sequence_array))])
            pos_info = np.stack(pos_info)

            # Combine one-hot encoding with positional information
            encoded_sequences = np.concatenate([encoded_sequences, pos_info], axis=-1)

        return encoded_sequences

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
