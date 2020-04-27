import uuid

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.ElementGenerator import ElementGenerator


class ElementDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None,
                 file_size: int = 1000, name: str = None):
        super().__init__()
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.element_generator = ElementGenerator(self._filenames)
        self.file_size = file_size
        self.element_ids = None
        self.name = name

    def get_data(self, batch_size: int = 1000):
        self._filenames.sort()
        self.element_generator.file_list = self._filenames
        return self.element_generator.build_element_generator(batch_size)

    def get_batch(self, batch_size: int = 1000):
        self._filenames.sort()
        self.element_generator.file_list = self._filenames
        return self.element_generator.build_batch_generator(batch_size)

    def get_filenames(self):
        return self._filenames

    def set_filenames(self, filenames):
        self._filenames = filenames
        self.element_generator = ElementGenerator(self._filenames)

    def get_example_count(self):
        return len(self.get_example_ids())

    def get_example_ids(self):
        if self.element_ids is None or (isinstance(self.element_ids, list) and len(self.element_ids) == 0):
            self.element_ids = []
            for element in self.get_data():
                self.element_ids.append(element.identifier)
        return self.element_ids

    def make_subset(self, example_indices, path, dataset_type: str):
        new_dataset = self.__class__(params=self.params, file_size=self.file_size)
        batch_filenames = self.element_generator.make_subset(example_indices, path, dataset_type, new_dataset.identifier)
        new_dataset.set_filenames(batch_filenames)
        return new_dataset
