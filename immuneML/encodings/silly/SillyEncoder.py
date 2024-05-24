import numpy as np

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.util.EncoderHelper import EncoderHelper


class SillyEncoder(DatasetEncoder):
    """
    This SillyEncoder class is a placeholder for a real encoder.
    It computes a set of random numbers as features for a given dataset.


    **Specification arguments:**

    - embedding_len (int): The number of random features to generate per example.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_silly_encoder:
                    Silly: # name of the class (without 'Encoder' suffix)
                        embedding_len: 5
    """

    def __init__(self, embedding_len: int = 5, name: str = None):
        super().__init__(name=name)
        self.embedding_len = embedding_len

    @staticmethod
    def build_object(dataset=None, **params):
        if 'embedding_len' in params:
            assert isinstance(params['embedding_len'], int), "SillyEncoder: embedding_len " \
                                                             "must be an integer"
            assert params['embedding_len'] >= 1, "SillyEncoder: embedding_len must be at least 1"

        if isinstance(dataset, SequenceDataset):
            return SillyEncoder(**params)
        else:
            raise ValueError("SillyEncoder is only defined for SequenceDatasets")

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        design_matrix = self._get_encoded_sequences(dataset)

        encoded_data = EncodedData(examples=design_matrix,
                                   example_ids=dataset.get_example_ids(),
                                   feature_names=[f"feature_{i}" for i in range(self.embedding_len)],
                                   # utility function for reformatting the labels
                                   labels=EncoderHelper.encode_dataset_labels(dataset,
                                                                              params.label_config,
                                                                              params.encode_labels),
                                   encoding=SillyEncoder.__name__)

        return SequenceDataset(filenames=dataset.get_filenames(), encoded_data=encoded_data,
                               labels=dataset.labels, file_size=dataset.file_size,
                               dataset_file=dataset.dataset_file)

    def _get_encoded_sequences(self, dataset: SequenceDataset) -> np.array:
        encoded_sequences = []

        for receptor_sequence in dataset.get_data():
            # examples of how information can be retrieved from the receptor_sequence:
            identifier = receptor_sequence.get_id()
            aa_seq = receptor_sequence.get_sequence()
            v_gene = receptor_sequence.get_attribute("v_gene")
            j_gene = receptor_sequence.get_attribute("j_gene")

            # in this encoding, ignore receptor_sequence, generate random features instead:
            random_encoding = np.random.rand(self.embedding_len)
            encoded_sequences.append(random_encoding)

        return np.array(encoded_sequences)