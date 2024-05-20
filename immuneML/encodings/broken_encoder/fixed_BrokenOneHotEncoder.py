import abc

import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams


# fix: BrokenOneHotSequenceEncoder -> BrokenOneHotEncoder
# fix: inherit from DatasetEncoder base class
class BrokenOneHotEncoder(DatasetEncoder):
    """
    One-hot encoding for sequence datasets. In one-hot encoding, each alphabet character
    (amino acid or nucleotide) is replaced by a sparse vector with one 1 and the rest zeroes.
    The position of the 1 represents the alphabet character.


    **Specification arguments:**

    - flatten (bool): whether to flatten the final onehot matrix to a 2-dimensional matrix [examples, other_dims_combined]
      This must be set to True when using onehot encoding in combination with scikit-learn ML methods (inheriting :py:obj:`~source.ml_methods.SklearnMethod.SklearnMethod`),
      such as :ref:`LogisticRegression`, :ref:`SVM`, :ref:`SVC`, :ref:`RandomForestClassifier` and :ref:`KNN`.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                one_hot:
                    BrokenOneHot:
                        flatten: False

    """

    def __init__(self, flatten: bool = True, name: str = None):
        super().__init__(name=name)
        self.flatten = flatten
        self.alphabet = EnvironmentSettings.get_sequence_alphabet()

    @staticmethod
    def build_object(dataset=None, **params):
        if "flatten" in params:
            ParameterValidator.assert_type_and_value(params["flatten"], bool, BrokenOneHotEncoder.__name__, "flatten")

        # fix: added return statement
        return BrokenOneHotEncoder(**params)

    def encode(self, dataset, params: EncoderParams):
        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):
        return (("dataset_identifier", dataset.identifier),
                ("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", BrokenOneHotEncoder.__name__),
                ("encoding_params", tuple(vars(self).items())))

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    # fix warning: encode_sequence_list -> _encode_sequence_list
    def _encode_sequence_list(self, sequences, pad_n_sequences, pad_sequence_len):
        char_array = np.array(sequences, dtype=str)
        char_array = char_array.view('U1').reshape((char_array.size, -1))

        n_sequences, sequence_len = char_array.shape

        sklearn_enc = SklearnOneHotEncoder(categories=[self.alphabet for i in range(sequence_len)], handle_unknown='ignore')
        encoded_data = sklearn_enc.fit_transform(char_array).toarray()

        encoded_data = np.pad(encoded_data, pad_width=((0, pad_n_sequences - n_sequences), (0, 0)))
        encoded_data = encoded_data.reshape((pad_n_sequences, sequence_len, len(self.alphabet)))
        encoded_data = np.pad(encoded_data, pad_width=((0, 0), (0, pad_sequence_len - sequence_len), (0, 0)))

        return encoded_data

    def _encode_new_dataset(self, dataset: SequenceDataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = SequenceDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          labels=dataset.labels,
                                          file_size=dataset.file_size, dataset_file=dataset.dataset_file)

        return encoded_dataset # fix: encoded_data -> encoded_dataset

    def _encode_data(self, dataset: SequenceDataset, params: EncoderParams):
        sequence_objs = [obj for obj in dataset.get_data()]
        sequences = [obj.get_sequence() for obj in sequence_objs]

        if any(seq is None for seq in sequences):
            raise ValueError(
                f"{BrokenOneHotEncoder.__name__}: sequence dataset {dataset.name} (id: {dataset.identifier}) contains empty sequences. "
                f"Please check that the dataset is imported correctly.")

        max_seq_len = max([len(seq) for seq in sequences])

        examples = self._encode_sequence_list(sequences, pad_n_sequences=len(sequence_objs), pad_sequence_len=max_seq_len)
        feature_names = self._get_nested_feature_names(max_seq_len)

        if self.flatten:
            examples = examples.reshape((len(sequence_objs), max_seq_len*len(self.alphabet)))
            feature_names = [item for sublist in feature_names for item in sublist]

        encoded_data = EncodedData(examples=examples,
                                   labels=EncoderHelper.encode_dataset_labels(dataset,
                                                                              params.label_config,
                                                                              params.encode_labels),
                                   example_ids=dataset.get_example_ids(),
                                   feature_names=feature_names,
                                   example_weights=dataset.get_example_weights(),
                                   encoding=BrokenOneHotEncoder.__name__)

        return encoded_data

    def _get_nested_feature_names(self, max_seq_len):
        return [[f"{pos}_{dim}" for dim in self.alphabet] for pos in range(max_seq_len)]

    # fix: remove overwritten method 'store'