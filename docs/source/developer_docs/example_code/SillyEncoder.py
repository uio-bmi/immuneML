import numpy as np

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class SillyEncoder(DatasetEncoder):
    """
    This SillyEncoder class is a placeholder for a real encoder.
    It computes a set of random numbers as features for a given dataset.

    **Specification arguments:**

    - random_seed (int): The random seed for generating random features.

    - embedding_len (int): The number of random features to generate per example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        my_silly_encoder:
            Silly: # name of the class (without 'Encoder' suffix)
                random_seed: 1
                embedding_len: 5
    """

    def __init__(self, random_seed: int, embedding_len: int, name: str = None):
        # The encoder name contains the user-defined name for the encoder. It may be used by some reports.
        super().__init__(name=name)

        # All user parameters are set here.
        # Default parameters must not be defined in the Encoder class, but in a default parameters file.
        self.random_seed = random_seed
        self.embedding_len = embedding_len

    @staticmethod
    def build_object(dataset=None, **params):
        # build_object is called early in the immuneML run, before the analysis takes place.
        # Its purpose is to fail early when a class is called incorrectly (checking parameters and dataset),
        # and provide user-friendly error messages.

        # ParameterValidator contains many utility functions for checking user parameters
        ParameterValidator.assert_type_and_value(params['random_seed'], int, SillyEncoder.__name__, 'random_seed', 1)
        ParameterValidator.assert_type_and_value(params['embedding_len'], int, SillyEncoder.__name__, 'embedding_len', 1, 100)

        # An error should be thrown if the dataset type is incompatible with the Encoder.
        # If different sub-classes are defined for each dataset type (e.g., OneHotRepertoireEncoder),
        # an instance of the dataset-specific class must be returned here.
        if isinstance(dataset, SequenceDataset):
            return SillyEncoder(**params)
        else:
            raise ValueError("SillyEncoder is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        np.random.seed(self.random_seed)

        # Generate the design matrix from the sequence dataset
        encoded_sequences = self._get_encoded_sequences(dataset, params)

        # EncoderHelper contains some utility functions, including this function for encoding the labels
        labels = EncoderHelper.encode_dataset_labels(dataset, params.label_config, params.encode_labels)

        # Each feature is represented by some meaningful name
        feature_names = [f"random_number_{i}" for i in range(self.embedding_len)]

        encoded_data = EncodedData(examples=encoded_sequences,
                                   labels=labels,
                                   example_ids=dataset.get_example_ids(),
                                   feature_names=feature_names,
                                   encoding=SillyEncoder.__name__)

        encoded_dataset = SequenceDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          labels=dataset.labels,
                                          file_size=dataset.file_size, dataset_file=dataset.dataset_file)

        return encoded_dataset

    def _get_encoded_sequences(self, dataset: SequenceDataset, params: EncoderParams) -> np.array:
        encoded_sequences = []

        for sequence in dataset.get_data(params.pool_size):
            # Different properties about the sequence, such as sequence.sequence_aa can be retrieved here
            # in this encoding, sequence information is ignored
            random_encoding = np.random.rand(self.embedding_len)
            encoded_sequences.append(random_encoding)

        return np.array(encoded_sequences)
