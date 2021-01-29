import abc


class Dataset:
    TRAIN = "train"
    TEST = "test"
    SUBSAMPLED = "subsampled"

    def __init__(self, encoded_data=None, name: str = None, identifier: str = None, labels: dict = None):
        self.encoded_data = encoded_data
        self.identifier = identifier
        self.name = name if name is not None else self.identifier
        self.labels = labels

    @abc.abstractmethod
    def make_subset(self, example_indices, path, dataset_type: str):
        pass

    @abc.abstractmethod
    def get_example_count(self):
        pass

    @abc.abstractmethod
    def get_data(self, batch_size: int = 1):
        pass

    @abc.abstractmethod
    def get_batch(self, batch_size: int = 1):
        pass

    @abc.abstractmethod
    def get_example_ids(self):
        pass

    @abc.abstractmethod
    def get_label_names(self):
        pass

    @abc.abstractmethod
    def clone(self):
        pass
