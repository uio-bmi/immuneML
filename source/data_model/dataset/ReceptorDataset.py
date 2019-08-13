import uuid

from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.ReceptorGenerator import ReceptorGenerator


class ReceptorDataset:

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None):
        self.params = params
        self.encoded_data = encoded_data
        self.id = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.receptor_generator = ReceptorGenerator(self._filenames)

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.receptor_generator.file_list = self._filenames
        return self.receptor_generator.build_generator(batch_size)

    def get_sequence_count(self):
        return self.receptor_generator.get_item_count()

    def get_metadata(self, field_names: list):
        raise NotImplementedError
