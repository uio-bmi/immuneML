from typing import List

from immuneML.simulation.Implanting import Implanting


class Simulation:

    def __init__(self, implantings: List[Implanting]):
        self.implantings = implantings

    def __str__(self):
        return ",\n".join(str(implanting) for implanting in self.implantings)
