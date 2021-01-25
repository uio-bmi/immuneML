# quality: gold
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation


class SequenceAnnotation:
    """
    Sequence Annotation class includes antigen-specific data (in experimental
    scenario) and implanted signals (in simulated scenario)
    """
    def __init__(self, implants: list = None, other: dict = None):
        self.implants = implants if implants is not None else []
        self.other = other if other is not None else {}

    def add_implant(self, implant: ImplantAnnotation):
        self.implants.append(implant)
