import math
import pickle
import uuid

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.receptor_sequence.SequenceGenerator import SequenceGenerator


class SequenceDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None,
                 file_size: int = 1000):
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.sequence_generator = SequenceGenerator(self._filenames)
        self.file_size = file_size

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.sequence_generator.file_list = self._filenames
        return self.sequence_generator.build_item_generator(batch_size)

    def get_batch(self, batch_size: int = 1000):
        self._filenames.sort()
        self.sequence_generator.file_list = self._filenames
        return self.sequence_generator.build_batch_generator(batch_size)

    def get_sequence_count(self):
        return self.sequence_generator.get_item_count()

    def get_filenames(self):
        return self._filenames

    def get_example_count(self):
        return self.get_sequence_count()

    def make_subset(self, example_indices, path):

        batch_size = 1000
        sequences = []
        file_count = 1
        batch_filenames = self._prepare_batch_filenames(len(example_indices), path)

        for index, batch in enumerate(self.get_batch(batch_size)):
            extracted_sequences = self._extract_sequences_from_batch(index, batch_size, batch, example_indices)
            sequences.extend(extracted_sequences)

            if len(sequences) >= self.file_size or len(sequences) == len(example_indices):
                self._store_sequences_to_file(batch_filenames[file_count-1], sequences[:self.file_size])
                file_count += 1
                sequences = sequences[self.file_size:]

        return SequenceDataset(params=self.params, filenames=batch_filenames, file_size=self.file_size)

    def _prepare_batch_filenames(self, example_count: int, path: str):
        batch_count = math.ceil(example_count / self.file_size)
        digits_count = len(str(batch_count))
        filenames = [path + "batch".join(["0" for i in range(digits_count-len(str(index)))]) + str(index) + ".pkl"
                     for index in range(digits_count)]
        return filenames

    def _store_sequences_to_file(self, path, sequences):
        with open(path, "wb") as file:
            pickle.dump(sequences, file)

    def _extract_sequences_from_batch(self, index, batch_size, batch, example_indices):
        upper_limit, lower_limit = (index + 1) * batch_size, index * batch_size
        batch_indices = [ind for ind in example_indices if lower_limit <= ind <= upper_limit]
        sequences = [batch[i - lower_limit] for i in batch_indices]
        return sequences
