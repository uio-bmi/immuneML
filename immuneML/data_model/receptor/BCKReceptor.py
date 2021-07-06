import json
from uuid import uuid4

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCKReceptor(Receptor):
    FIELDS = {'heavy': object, 'kappa': object, 'identifier': str, 'metadata': dict, 'version': str}
    version = "1"

    @classmethod
    def create_from_record(cls, record):
        if 'version' in record.dtype.names and record['version'] == BCKReceptor.version:

            heavy_record = record[['heavy_' + name for name in ReceptorSequence.get_record_names()]]
            heavy_record.dtype.names = ReceptorSequence.get_record_names()

            kappa_record = record[['kappa_' + name for name in ReceptorSequence.get_record_names()]]
            kappa_record.dtype.names = ReceptorSequence.get_record_names()

            return BCKReceptor(heavy=ReceptorSequence.create_from_record(heavy_record),
                               kappa=ReceptorSequence.create_from_record(kappa_record),
                               identifier=record['identifier'], metadata=json.loads(record['metadata']))
        else:
            raise NotImplementedError(f"Supported ({BCKReceptor.version}) and available version differ, but there is no converter available.")

    @classmethod
    def get_record_names(cls):
        return ['heavy_' + name for name in ReceptorSequence.get_record_names()] \
               + ['kappa_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['heavy', 'kappa']]

    def __init__(self, heavy: ReceptorSequence = None, kappa: ReceptorSequence = None, metadata: dict = None, identifier: str = None):
        self.heavy = heavy
        self.kappa = kappa
        self.metadata = metadata
        self.identifier = uuid4().hex if identifier is None else identifier

    def get_chains(self):
        return ["heavy", "kappa"]

    def get_attribute(self, name: str):
        raise NotImplementedError
