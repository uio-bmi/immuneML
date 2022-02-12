from dataclasses import dataclass

from immuneML.simulation.implants.MotifInstance import MotifInstance


@dataclass
class ImplantAnnotation:
    signal_id: str = None
    motif_id: str = None
    motif_instance: MotifInstance = None
    position: int = None

    def __str__(self):
        return str({key: str(val) for key, val in vars(self).items()})
