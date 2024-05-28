import logging


class Label:

    def __init__(self, name: str, values: list, auxiliary_label_names: list = None, positive_class=None):
        self.name = name
        self._values = values
        self.auxiliary_label_names = auxiliary_label_names
        self._positive_class = positive_class

    def __str__(self):
        return f"label {self.name} ({', '.join([str(val) for val in self.values])})"

    def __eq__(self, other):
        if not isinstance(other, Label):
            return False

        return self.name == other.name and self.values == other.values and \
            self.positive_class == other.positive_class and self.auxiliary_label_names == other.auxiliary_label_names

    @property
    def positive_class(self):
        '''
        Ensures the same class is always returned as the 'positive class', even when it was not explicitly set.
        '''
        if self._positive_class is not None:
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
        '''
        Returns the label values and makes sure the positive class is listed last
        This is needed for compatibility with `~immuneML.ml_methods.util.Util.binarize_label_classes`
        '''
        if self._positive_class is not None:
            negative_classes = sorted(self._values)
            negative_classes.remove(self._positive_class)
            return negative_classes + [self._positive_class]
        else:
            return sorted(self._values)

    def get_binary_negative_class(self):
        '''
        Ensures the correct 'negative class' is returned when using the Label for binary classification.
        '''
        assert len(self._values) == 2, f"Label: binary negative class was requested for label {self.name} but this label contains {len(self._values)} classes: {self.values}"
        return [val for val in self._values if val != self.positive_class][0]

    def get_desc_for_storage(self):
        '''
        Method to call when storing a label to YAML format
        '''
        return {key.lstrip("_"): value for key, value in vars(self).items()}

