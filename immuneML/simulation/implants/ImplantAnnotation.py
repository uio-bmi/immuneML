from dataclasses import dataclass


@dataclass
class ImplantAnnotation:
    signal_id: str = None
    motif_id: str = None
    motif_instance: str = None
    position: int = None

    def __str__(self):
        return str({key: str(val) for key, val in vars(self).items()})
