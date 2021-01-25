import abc
import copy
from uuid import uuid4

from immuneML.data_model.DatasetItem import DatasetItem


class Receptor(DatasetItem):

    @abc.abstractmethod
    def get_chains(self):
        pass

    def clone(self):
        copied_element = copy.deepcopy(self)
        copied_element.identifier = str(uuid4().hex)
        return copied_element

    def get_chain(self, chain: str):
        return getattr(self, chain)
