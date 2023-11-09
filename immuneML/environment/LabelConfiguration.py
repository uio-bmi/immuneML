# quality: gold
import logging
import warnings
from typing import List

from immuneML.environment.Label import Label
from immuneML.util.ParameterValidator import ParameterValidator


class LabelConfiguration:
    """
    Class that encapsulates labels and transformers for the labels.
    Supports two types of labels: CLASSIFICATION and REGRESSION (as defined in LabelType class)
    """

    def __init__(self, labels: list = None):

        assert labels is None or all(isinstance(label, Label) for label in labels), \
            "LabelConfiguration: all labels should be instances of Label class."

        self._labels = {label.name: label for label in labels} if labels is not None else {}

    def add_label(self, label_name: str, values: list = None, auxiliary_labels: list = None, positive_class=None):

        vals = list(values) if values else None

        if label_name in self._labels and self._labels[label_name] is not None and len(self._labels[label_name]) > 0:
            warnings.warn("Label " + label_name + " has already been set. Overriding existing values...", Warning)

        if positive_class is not None:
            if all(isinstance(val, str) for val in vals) and not isinstance(positive_class, str):
                positive_class = str(positive_class)
            ParameterValidator.assert_in_valid_list(positive_class, vals, Label.__name__, 'positive_class')
        else:
            positive_class = self._get_default_positive_class(vals)
            logging.info(f"LabelConfiguration: No positive label class was set. "
                         f"Setting default positive class '{positive_class}' for label {label_name}")

        self._labels[label_name] = Label(label_name, vals, auxiliary_labels, positive_class)

    def _get_default_positive_class(self, classes):
        """Returns the default positive class when a class pair is given where the positive class is obvious (0,
        1; true, false)"""

        if len(classes) != 2:
            return None
        if classes[0] == True and classes[1] == False:
            return classes[0]
        if classes[1] == True and classes[0] == False:
            return classes[1]

        for positive_str, negative_str in [("1", "0"), ("true", "false"), ("positive", "negative"), ("+", "-")]:
            if set(classes) == {positive_str, negative_str}:
                return positive_str
            if set(classes) == {positive_str.upper(), negative_str.upper()}:
                return positive_str.upper()
            if set(classes) == {positive_str.title(), negative_str.title()}:
                return positive_str.title()

        return sorted(classes)[0]

    def get_labels_by_name(self):
        return sorted(list(self._labels.keys()))

    def get_label_values(self, label_name: str):
        assert label_name in self._labels, label_name + " is not in the list of labels, so there is no information on the values."
        return self._labels[label_name].values

    def get_label_count(self):
        return len(self._labels.keys())

    def get_auxiliary_labels(self, label_name: str):
        return self._labels[label_name].auxiliary_label_names

    def get_label_object(self, label_name: str) -> Label:
        return self._labels[label_name]

    def get_label_objects(self) -> List[Label]:
        return list(self._labels.values())
