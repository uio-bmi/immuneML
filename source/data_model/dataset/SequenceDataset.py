import uuid

from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.receptor_sequence.SequenceGenerator import SequenceGenerator


class SequenceDataset:

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None):
        self.params = params
        self.encoded_data = encoded_data
        self.id = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.sequence_generator = SequenceGenerator(self._filenames)

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.sequence_generator.file_list = self._filenames
        return self.sequence_generator.build_generator(batch_size)

    def get_sequence_count(self):
        return self.sequence_generator.get_item_count()

    def get_metadata(self, field_names: list):
        raise NotImplementedError

    def get_filenames(self):
        return self._filenames