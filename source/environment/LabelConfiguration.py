# quality: gold
import warnings


class LabelConfiguration:
    """
    Class that encapsulates labels and transformers for the labels.
    Supports two types of labels: CLASSIFICATION and REGRESSION (as defined in LabelType class)
    """
    # TODO: add label config object to dataset.params
    def __init__(self, labels: dict = None):

        assert labels is None or all(isinstance(labels[key], list) for key in labels), \
            "LabelConfiguration: labels dict does not have the format {label_name1: [possible_val1, possible_val2], " \
            "label_name2: [possible_val3, possible_val4]}."

        self._labels = labels if labels is not None else {}

    def add_label(self, label: str, values: list = None):

        vals = list(values) if values else None

        if label in self._labels and self._labels[label] is not None and len(self._labels[label]) > 0:
            warnings.warn("Label " + label + " has already been set. Overriding existing values...", Warning)

        self._labels[label] = vals

    def get_labels_by_name(self):
        return sorted(list(self._labels.keys()))

    def get_label_values(self, label: str):
        assert label in self._labels, label + " is not in the list of labels, so there is no information on the values."
        return self._labels[label]

    def get_label_count(self):
        return len(self._labels.keys())
