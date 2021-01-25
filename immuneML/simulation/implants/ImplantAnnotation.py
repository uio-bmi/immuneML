from immuneML.simulation.implants.MotifInstance import MotifInstance


class ImplantAnnotation:

    def __init__(self, signal_id=None, motif_id=None, motif_instance: MotifInstance = None, position=None):
        self.signal_id = signal_id
        self.motif_id = motif_id
        self.motif_instance = motif_instance
        self.position = position

    def __str__(self):
        return "{" + f"'signal_id': '{self.signal_id}', 'motif_id': '{self.motif_id}', 'motif_instance': '{str(self.motif_instance)}', " \
                     f"'position': '{self.position}'" + "}"
