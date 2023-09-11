import abc
import copy
from itertools import chain
from typing import List
from uuid import uuid4

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.util.NumpyHelper import NumpyHelper


class Receptor(DatasetItem):
    FIELDS = {}

    def __init__(self, identifier: str = None, metadata: dict = None):
        self.metadata = metadata
        self.identifier = identifier if identifier is not None else uuid4().hex

    @abc.abstractmethod
    def get_chains(self):
        pass

    @classmethod
    def create_from_record(cls, **kwargs):

        chains = cls().get_chains()

        chain1_record = {key.replace(f'{chains[0]}_', ''): val for key, val in kwargs.items() if
                         key.startswith(chains[0])}

        chain2_record = {key.replace(f'{chains[1]}_', ''): val for key, val in kwargs.items() if
                         key.startswith(chains[1])}

        assert chain1_record['cell_id'] == chain2_record['cell_id'], (chain1_record['cell_id'], chain2_record['cell_id'])
        if 'identifier' in chain1_record:
            assert chain1_record['identifier'] == chain2_record['identifier'], (chain1_record['identifier'], chain2_record['identifier'])
            identifier = kwargs[f'{chains[0]}_identifier']
        else:
            identifier = kwargs[f'{chains[0]}_cell_id']

        return cls(**{chains[0]: ReceptorSequence.create_from_record(**chain1_record),
                      chains[1]: ReceptorSequence.create_from_record(**chain2_record),
                      'identifier': identifier,
                      'metadata': {key.replace(f'{chains[0]}_', '').replace(f'{chains[1]}_', ''): val
                                   for key, val in kwargs.items()
                                   if key.replace(f'{chains[0]}_', '').replace(f'{chains[1]}_', '') not in ReceptorSequence.FIELDS.keys() and key.replace(f'{chains[0]}_', '').replace(f'{chains[1]}_', '') not in vars(SequenceMetadata()).keys()}})

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
                 + [NumpyHelper.get_numpy_representation(getattr(self, name)) for name in self.FIELDS if
                    name not in chains]

        return record

    def get_id(self):
        return self.identifier

    def get_attribute(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        if self.metadata is not None and name in self.metadata.keys():
            return self.metadata[name].to_string() if hasattr(self.metadata[name], 'to_string') else self.metadata[name]
        else:
            return [self.get_chain(ch).get_attribute(name) for ch in self.get_chains()]

    def get_all_attribute_names(self) -> List[str]:
        names = list(self.metadata.keys()) if self.metadata is not None else [] + ['cell_id']
        names += ['identifier']
        names += list(set(chain.from_iterable([self.get_chain(ch).get_all_attribute_names() for ch in self.get_chains()])))
        return names

    def __repr__(self):
        return self.__class__.__name__ + f"(identifier={self.identifier})"
