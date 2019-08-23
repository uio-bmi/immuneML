import abc


class Dataset:

    TRAIN = "train"
    TEST = "test"

    @abc.abstractmethod
    def make_subset(self, example_indices, path):
        pass

    @abc.abstractmethod
    def get_example_count(self):
        pass

    @abc.abstractmethod
    def get_data(self, batch_size: int):
        pass

    @abc.abstractmethod
    def get_batch(self, batch_size: int):
        pass
