import abc


class Dataset:

    TRAIN = "train"
    TEST = "test"

    def __init__(self):
        self.encoded_data = None
        self.name = None
        self.identifier = None

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
    def clone(self):
        pass
