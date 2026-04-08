import logging
import re

_INT_PATTERN = re.compile(r'^[+-]?\d+$')


def _is_float_str(s: str) -> bool:
    s = s.lstrip('+-')
    parts = s.split('.')
    return len(parts) == 2 and (parts[0] == '' or parts[0].isdigit()) and parts[1].isdigit()


def infer_label_types(values: list) -> list:
    """
    Convert a list of string label values to their natural numeric type when all values match.
    Priority: int > float > unchanged.
    """
    if values is None or len(values) == 0 or not all(isinstance(v, str) for v in values):
        return values
    if all(_INT_PATTERN.fullmatch(v) for v in values):
        return [int(v) for v in values]
    if all(_is_float_str(v) for v in values):
        return [float(v) for v in values]
    return values


class Label:

    def __init__(self, name: str, values: list = None, auxiliary_label_names: list = None, positive_class=None):
        self.name = name
        self._values = values
        self.auxiliary_label_names = auxiliary_label_names
        self._positive_class = positive_class

    def __str__(self):
        return f"label {self.name} ({', '.join([str(val) for val in self.values])})"

    def __eq__(self, other):
        if not isinstance(other, Label):
            return False

        return self.name == other.name and sorted(self.values) == sorted(other.values) and \
            self.positive_class == other.positive_class and self.auxiliary_label_names == other.auxiliary_label_names

    @property
    def positive_class(self):
        """
        Ensures the same class is always returned as the 'positive class', even when it was not explicitly set.
        When a type mismatch exists between the stored positive_class and the actual label values (e.g., int 1
        vs string '1' due to pandas loading data as strings), the value is coerced to match the values' type.
        """
        if self._positive_class is not None:
            if self._values is not None and self._positive_class not in self._values:
                coerced = next((v for v in self._values if str(v) == str(self._positive_class)), None)
                if coerced is not None:
                    self._positive_class = coerced
            return self._positive_class
        else:
            assert self._values is not None, f"Label: cannot access positive class for label {self.name} " \
                                             f"when neither 'positive_class' nor 'values' are set."

            positive_class = sorted(self._values)[0]
            logging.info(f"Label: No positive class was set for label {self.name}. "
                         f"Assuming default positive class '{positive_class}'.")

            return positive_class

    @property
    def values(self):
        """
        Make sure the positive class is listed last
        This is needed for compatibility with `~immuneML.ml_methods.util.Util.binarize_label_classes`
        """
        if self._positive_class is not None:
            negative_classes = sorted(self._values)
            negative_classes.remove(self._positive_class)
            return negative_classes + [self._positive_class]
        else:
            return sorted(self._values)

    def get_binary_negative_class(self):
        """
        Ensures the correct 'negative class' is returned when using the Label for binary classification.
        """
        assert len(self._values) == 2, f"Label: binary negative class was requested for label {self.name} but this label contains {len(self._values)} classes: {self.values}"
        return [val for val in self._values if val != self.positive_class][0]

    def get_desc_for_storage(self):
        """
        Method to call when storing a label to YAML format
        """
        return {key.lstrip("_"): value for key, value in vars(self).items()}



