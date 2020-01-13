# quality: gold
import warnings

from source.environment.Label import Label


class LabelConfiguration:
    """
    Class that encapsulates labels and transformers for the labels.
    Supports two types of labels: CLASSIFICATION and REGRESSION (as defined in LabelType class)
    """
    # TODO: add label config object to dataset.params
    def __init__(self, labels: list = None):

        assert labels is None or all(isinstance(label, Label) for label in labels), \
            "LabelConfiguration: all labels should be instances of Label class."

        self._labels = {label.name: label for label in labels} if labels is not None else {}

    def add_label(self, label: str, values: list = None, auxiliary_labels: list = None):

        vals = list(values) if values else None

        if label in self._labels and self._labels[label] is not None and len(self._labels[label]) > 0:
            warnings.warn("Label " + label + " has already been set. Overriding existing values...", Warning)

        self._labels[label] = Label(label, vals, auxiliary_labels)

    def get_labels_by_name(self):
        return sorted(list(self._labels.keys()))

    def get_label_values(self, label: str):
        assert label in self._labels, label + " is not in the list of labels, so there is no information on the values."
        return self._labels[label].values

    def get_label_count(self):
        return len(self._labels.keys())

    def get_auxiliary_labels(self, label: str):
        return self._labels[label].auxiliary_label_names

    def get_label_object(self, label: str) -> Label:
        return self._labels[label]
