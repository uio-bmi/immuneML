import uuid

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.util.ItemGenerator import ItemGenerator


class ItemDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None,
                 file_size: int = 1000):
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.item_generator = ItemGenerator(self._filenames)
        self.file_size = file_size

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.item_generator.file_list = self._filenames
        return self.item_generator.build_item_generator(batch_size)

    def get_batch(self, batch_size: int = 1000):
        self._filenames.sort()
        self.item_generator.file_list = self._filenames
        return self.item_generator.build_batch_generator(batch_size)

    def get_filenames(self):
        return self._filenames

    def get_example_count(self):
        return self.item_generator.get_item_count()

    def make_subset(self, example_indices, path):
        batch_filenames = self.item_generator.make_subset(example_indices, path)
        return self.__class__(params=self.params, filenames=batch_filenames, file_size=self.file_size)
