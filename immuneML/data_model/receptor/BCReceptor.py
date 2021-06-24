import json
from uuid import uuid4

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCReceptor(Receptor):
    FIELDS = {'heavy': object, 'light': object, 'identifier': str, 'metadata': dict, 'version': str}
    version = "1"

    @classmethod
    def create_from_record(cls, record):
        if 'version' in record.dtype.names and record['version'] == BCReceptor.version:

            heavy_record = record[['heavy_' + name for name in ReceptorSequence.get_record_names()]]
            heavy_record.dtype.names = ReceptorSequence.get_record_names()

            light_record = record[['light_' + name for name in ReceptorSequence.get_record_names()]]
            light_record.dtype.names = ReceptorSequence.get_record_names()

            return BCReceptor(heavy=ReceptorSequence.create_from_record(heavy_record),
                              light=ReceptorSequence.create_from_record(light_record),
                              identifier=record['identifier'], metadata=json.loads(record['metadata']))
        else:
            raise NotImplementedError(f"Supported ({BCReceptor.version}) and available version differ, but there is no converter available.")

    @classmethod
    def get_record_names(cls):
        return ['heavy_' + name for name in ReceptorSequence.get_record_names()] \
               + ['light_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['heavy', 'light']]

    def __init__(self, heavy: ReceptorSequence = None, light: ReceptorSequence = None, metadata: dict = None,
                 identifier: str = None):
        self.heavy = heavy
        self.light = light
        self.metadata = metadata
        self.identifier = uuid4().hex if identifier is None else identifier

    def get_chains(self):
        return ["heavy", "light"]

    def get_attribute(self, name: str):
        raise NotImplementedError
