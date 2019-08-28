import uuid

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.ReceptorGenerator import ReceptorGenerator


class ReceptorDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None):
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.receptor_generator = ReceptorGenerator(self._filenames)

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.receptor_generator.file_list = self._filenames
        return self.receptor_generator.build_item_generator(batch_size)

    def get_batch(self, batch_size: int = 1000):
        self._filenames.sort()
        self.receptor_generator.file_list = self._filenames
        return self.receptor_generator.build_batch_generator(batch_size)

    def get_receptor_count(self):
        return self.receptor_generator.get_item_count()

    def get_example_count(self):
        return self.get_receptor_count()

    def make_subset(self, example_indices, path):
        raise NotImplementedError
