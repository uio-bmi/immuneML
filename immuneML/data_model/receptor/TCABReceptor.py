import json
from uuid import uuid4

import numpy as np

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCABReceptor(Receptor):
    FIELDS = {'alpha': object, 'beta': object, 'identifier': str, 'metadata': dict, 'version': str}
    version = "1"

    @classmethod
    def create_from_record(cls, record: np.void):
        if 'version' in record.dtype.names and record['version'] == TCABReceptor.version:

            alpha_record = record[['alpha_' + name for name in ReceptorSequence.get_record_names()]]
            alpha_record.dtype.names = ReceptorSequence.get_record_names()

            beta_record = record[['beta_' + name for name in ReceptorSequence.get_record_names()]]
            beta_record.dtype.names = ReceptorSequence.get_record_names()

            return TCABReceptor(alpha=ReceptorSequence.create_from_record(alpha_record),
                                beta=ReceptorSequence.create_from_record(beta_record),
                                identifier=record['identifier'], metadata=json.loads(record['metadata']))
        else:
            raise NotImplementedError(f"Supported ({TCABReceptor.version}) and available version differ, but there is no converter available.")

    def __init__(self, alpha: ReceptorSequence = None, beta: ReceptorSequence = None, metadata: dict = None, identifier: str = None):
        self.alpha = alpha
        self.beta = beta
        self.metadata = metadata
        self.identifier = uuid4().hex if identifier is None else identifier

    def get_chains(self):
        return ["alpha", "beta"]

    @classmethod
    def get_record_names(cls):
        return ['alpha_' + name for name in ReceptorSequence.get_record_names()] \
               + ['beta_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['alpha', 'beta']]

    def get_attribute(self, name: str):
        raise NotImplementedError
