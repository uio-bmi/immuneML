import json
from uuid import uuid4

import numpy as np

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCGDReceptor(Receptor):
    FIELDS = {'gamma': object, 'delta': object, 'identifier': str, 'metadata': dict, 'version': str}
    version = "1"

    @classmethod
    def create_from_record(cls, record: np.void):
        if 'version' in record.dtype.names and record['version'] == TCGDReceptor.version:

            gamma_record = record[['gamma_' + name for name in ReceptorSequence.get_record_names()]]
            gamma_record.dtype.names = ReceptorSequence.get_record_names()

            delta_record = record[['delta_' + name for name in ReceptorSequence.get_record_names()]]
            delta_record.dtype.names = ReceptorSequence.get_record_names()

            return TCGDReceptor(gamma=ReceptorSequence.create_from_record(gamma_record),
                                delta=ReceptorSequence.create_from_record(delta_record),
                                identifier=record['identifier'], metadata=json.loads(record['metadata']))
        else:
            raise NotImplementedError(f"Supported ({TCGDReceptor.version}) and available version differ, but there is no converter available.")

    @classmethod
    def get_record_names(cls):
        return ['gamma_' + name for name in ReceptorSequence.get_record_names()] \
               + ['delta_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['gamma', 'delta']]

    def __init__(self, gamma: ReceptorSequence = None, delta: ReceptorSequence = None, metadata: dict = None, identifier: str = None):

        self.gamma = gamma
        self.delta = delta
        self.metadata = metadata
        self.identifier = identifier if identifier is not None else uuid4().hex

    def get_chains(self):
        return ["gamma", "delta"]

    def get_attribute(self, name: str):
        raise NotImplementedError
