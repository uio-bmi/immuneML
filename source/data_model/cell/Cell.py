from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.ReceptorList import ReceptorList


class Cell:

    def __init__(self, receptors: ReceptorList, metadata: dict = None):
        assert all(isinstance(receptor, Receptor) for receptor in receptors), \
            "Cell: the receptor list includes non-receptor object instances."

        self.receptors = receptors
        self.metadata = metadata
