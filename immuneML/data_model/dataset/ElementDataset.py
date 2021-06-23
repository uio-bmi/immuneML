from pathlib import Path
from uuid import uuid4

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.ElementGenerator import ElementGenerator


class ElementDataset(Dataset):
    """
    This is the base class for ReceptorDataset and SequenceDataset which implements all the functionality for both classes. The only difference between
    these two classes is whether paired or single chain data is stored.
    """

    def __init__(self, labels: dict = None, encoded_data: EncodedData = None, filenames: list = None, identifier: str = None,
                 file_size: int = 50000, name: str = None, element_class_name: str = None, element_ids: list = None):
        super().__init__()
        self.labels = labels
        self.encoded_data = encoded_data
        self.identifier = identifier if identifier is not None else uuid4().hex
        self.filenames = sorted(filenames) if filenames is not None else []
        self.filenames = [Path(filename) for filename in self.filenames]
        self.element_generator = ElementGenerator(self.filenames, file_size, element_class_name)
        self.file_size = file_size
        self.element_ids = element_ids
        self.name = name
        self.element_class_name = element_class_name

    def get_data(self, batch_size: int = 10000):
        self.filenames.sort()
        self.element_generator.file_list = self.filenames
        return self.element_generator.build_element_generator()

    def get_batch(self, batch_size: int = 10000):
        self.filenames.sort()
        self.element_generator.file_list = self.filenames
        return self.element_generator.build_batch_generator()

    def get_filenames(self):
        return self.filenames

    def set_filenames(self, filenames):
        self.filenames = filenames

    def get_example_count(self):
        return len(self.get_example_ids())

    def get_example_ids(self):
        if self.element_ids is None or (isinstance(self.element_ids, list) and len(self.element_ids) == 0):
            self.element_ids = []
            for element in self.get_data():
                self.element_ids.append(str(element.identifier))
        return self.element_ids

    def make_subset(self, example_indices, path, dataset_type: str):
        """
        Creates a new dataset object with only those examples (receptors or receptor sequences) available which were given by index in example_indices argument.

        Args:
            example_indices (list): a list of indices of examples (receptors or receptor sequences) to use in the new dataset
            path (Path): a path where to store the newly created dataset
            dataset_type (str): a type of the dataset used as a part of the name of the resulting dataset; the values are defined as constants in :py:obj:`~immuneML.data_model.dataset.Dataset.Dataset`

        Returns:

            a new dataset object (ReceptorDataset or SequenceDataset, as the original dataset) which includes only the examples specified under example_indices

        """
        new_dataset = self.__class__(labels=self.labels, file_size=self.file_size, element_class_name=self.element_generator.element_class_name)
        batch_filenames = self.element_generator.make_subset(example_indices, path, dataset_type, new_dataset.identifier)
        new_dataset.set_filenames(batch_filenames)
        new_dataset.name = f"{self.name}_split_{dataset_type.lower()}"
        return new_dataset

    def get_label_names(self):
        """Returns the list of metadata fields which can be used as labels"""
        return [label for label in list(self.labels.keys()) if label not in ['region_type', 'receptor_chains', 'organism']]

    def clone(self, keep_identifier: bool = False):
        raise NotImplementedError
