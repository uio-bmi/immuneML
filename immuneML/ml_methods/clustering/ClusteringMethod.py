import abc

from immuneML.data_model.dataset.Dataset import Dataset


class ClusteringMethod:

    def __init__(self, name: str = None):
        self.name = name

    @abc.abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def predict(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def transform(self, dataset: Dataset):
        pass


def get_data_for_clustering(dataset: Dataset):
    if dataset.encoded_data.dimensionality_reduced_data is not None:
        return dataset.encoded_data.dimensionality_reduced_data
    else:
        return dataset.encoded_data.examples
