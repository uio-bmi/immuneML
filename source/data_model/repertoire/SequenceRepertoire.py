# quality: gold

import collections
import uuid


class SequenceRepertoire:

    def __init__(self, sequences: collections.Iterable = None, metadata=None, identifier: str = None):
        self.sequences = sequences
        self.metadata = metadata
        self.identifier = identifier if identifier is not None else str(uuid.uuid1())

    def get_identifier(self):
        return self.identifier
