# quality: gold

from source.data_model.metadata.Label import Label
from source.data_model.metadata.Sample import Sample
from source.simulation.implants.ImplantAnnotation import ImplantAnnotation


class RepertoireMetadata:
    """
    Includes all metadata for a repertoire:
        - sample object describing experiment the data came from
        - list of labels to be used for machine learning
        - list of implants (signal, motif and motif instance) in simulation scenario
    """

    def __init__(self, sample: Sample = None, labels: list = None):

        if labels is not None:
            assert all([isinstance(label, Label) for label in labels])

        self.sample = sample
        self.labels = labels if labels is not None else []
        self.other_metadata = {}
        self.implants = []

    def add_implant(self, implant: ImplantAnnotation):
        self.implants.append(implant)

    def add_to_other_metadata(self, name, value):
        self.other_metadata[name] = value

    def add_sample(self, sample: Sample):
        self.sample = sample

    def add_label(self, label: Label):
        self.labels.append(label)
        self.labels.sort(key=lambda x: x.name)

    def add_labels(self, labels: list):
        assert all([isinstance(label, Label) for label in labels])
        self.labels.extend(labels)
        self.labels.sort(key=lambda x: x.name)

    def __str__(self):
        return str(self.sample.id) + "\t Labels:" + "|".join([l.value for l in self.labels])

