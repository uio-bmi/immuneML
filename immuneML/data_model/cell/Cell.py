from typing import List

from immuneML.data_model.receptor.Receptor import Receptor


class Cell:

    def __init__(self, receptors: List[Receptor], metadata: dict = None):
        assert all(isinstance(receptor, Receptor) for receptor in receptors), \
            "Cell: the receptor list includes non-receptor object instances."

        self.receptors = receptors
        self.metadata = metadata
