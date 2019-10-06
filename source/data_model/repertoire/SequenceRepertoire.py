# quality: gold

import collections
import uuid

from source.data_model.DatasetItem import DatasetItem


class SequenceRepertoire(DatasetItem):

    def __init__(self, sequences: collections.Iterable = None, metadata=None, identifier: str = None):
        self.sequences = sequences
        self.metadata = metadata
        self.identifier = identifier if identifier is not None else str(uuid.uuid1())

    def get_identifier(self):
        return self.identifier

    def get_attribute(self, name: str):
        raise NotImplementedError
