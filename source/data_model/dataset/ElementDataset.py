import uuid

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.ElementGenerator import ElementGenerator


class ElementDataset(Dataset):

    def __init__(self, params: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None,
                 file_size: int = 1000):
        self.params = params
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid.uuid1()
        self._filenames = sorted(filenames) if filenames is not None else []
        self.element_generator = ElementGenerator(self._filenames)
        self.file_size = file_size
        self.element_ids = None

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

    def get_example_count(self):
        return self.element_generator.get_element_count()

    def get_example_ids(self):
        if self.element_ids is None:
            self.element_ids = []
            for element in self.get_data():
                self.element_ids.append(element.identifier)
        return self.element_ids

    def make_subset(self, example_indices, path):
        batch_filenames = self.element_generator.make_subset(example_indices, path)
        return self.__class__(params=self.params, filenames=batch_filenames, file_size=self.file_size)
