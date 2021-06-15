import abc
import copy
from uuid import uuid4

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.util.NumpyHelper import NumpyHelper


class Receptor(DatasetItem):

    FIELDS = {}

    @abc.abstractmethod
    def get_chains(self):
        pass

    @classmethod
    @abc.abstractmethod
    def create_from_record(cls, record):
        pass

    @classmethod
    @abc.abstractmethod
    def get_record_names(cls):
        pass

    def clone(self):
        copied_element = copy.deepcopy(self)
        copied_element.identifier = str(uuid4().hex)
        return copied_element

    def get_chain(self, chain: str):
        return getattr(self, chain)

    def get_record(self):
        chains = self.get_chains()
        record = self.get_chain(chains[0]).get_record() + self.get_chain(chains[1]).get_record() \
                 + [NumpyHelper.get_numpy_representation(getattr(self, name)) for name in self.FIELDS if name not in chains]

        return record
