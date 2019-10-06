import abc

from source.data_model.DatasetItem import DatasetItem


class Receptor(DatasetItem):

    @abc.abstractmethod
    def get_chains(self):
        pass

    def get_chain(self, chain: str):
        return getattr(self, chain)
