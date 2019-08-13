from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCReceptor(Receptor):

    def __init__(self, heavy: ReceptorSequence = None, light: ReceptorSequence = None, metadata: dict = None, id: str = None):
        self.heavy = heavy
        self.light = light
        self.metadata = metadata
        self.id = id
