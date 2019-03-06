# quality: gold

from source.data_model.metadata.Sample import Sample
from source.simulation.implants.ImplantAnnotation import ImplantAnnotation


class RepertoireMetadata:
    """
    Includes all metadata for a repertoire:
        - sample object describing experiment the data came from in case all sequences came from the same sample
        - list of implants (signal, motif and motif instance) in simulation scenario
    """

    def __init__(self, sample: Sample = None, custom_params: dict = None):

        self.sample = sample
        self.custom_params = custom_params if custom_params is not None else {}
        self.implants = []

    def add_implant(self, implant: ImplantAnnotation):
        self.implants.append(implant)

    def add_sample(self, sample: Sample):
        self.sample = sample

    def __str__(self):
        return str(self.sample.id) + "\t Labels:" + "|".join([c.value for c in self.custom_params])

