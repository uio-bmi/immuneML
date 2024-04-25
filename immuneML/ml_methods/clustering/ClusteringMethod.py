import abc

from immuneML.data_model.dataset.Dataset import Dataset


class ClusteringMethod:
    '''
    .. note::

        This is an experimental feature

    Clustering methods are algorithms which can be used to cluster repertoires, receptors or
    sequences without using external label information (such as disease or antigen binding state)

    These methods can be used in the :ref:`Clustering` instruction.

    '''
    DOCS_TITLE = "Clustering methods"

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
