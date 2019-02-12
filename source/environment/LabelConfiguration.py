# quality: gold
import warnings

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from source.environment.LabelType import LabelType


class LabelConfiguration:
    """
    Class that encapsulates labels and transformers for the labels.
    Supports two types of labels: CLASSIFICATION and REGRESSION (as defined in LabelType class)
    """
    # TODO: add label config object to dataset.params
    def __init__(self):
        self.__labels = {}
        self.__label_binarizers = {}
        self.__label_encoders = {}

    def add_label(self, label: str, values: list, label_type: LabelType = LabelType.CLASSIFICATION):

        if label in self.__labels and self.__labels[label] is not None and len(self.__labels[label]) > 0:
            warnings.warn("Label " + label + " has already been set. Overriding existing values...", Warning)

        self.__labels[label] = values

        if label_type == LabelType.CLASSIFICATION:

            label_binarizer = LabelBinarizer()
            label_binarizer.fit(values)
            self.__label_binarizers[label] = label_binarizer

            label_encoder = LabelEncoder()
            label_encoder.fit(values)
            self.__label_encoders[label] = label_encoder

    def get_labels_by_name(self):
        return sorted(self.__labels.keys())

    def get_label_values(self, label: str):
        assert label in self.__labels, label + " is not in the list of labels, so there is no information on the values."
        return self.__labels[label]

    def get_label_binarizer(self, label: str):
        assert label in self.__labels, label + " is not in the list of labels, so there is no binarizer."
        assert label in self.__label_binarizers, label + " is in the list of labels, but there is no binerizer. " \
                                                        "The reason could be that the label was not added using " \
                                                        "add_label() or that it contains continuous values " \
                                                        "(LabelType.REGRESSION was specified when called add_label())."
        return self.__label_binarizers[label]

    def get_label_encoder(self, label: str):
        assert label in self.__labels, label + " is not in the list of labels, so there is no binarizer."
        assert label in self.__label_encoders, label + " is in the list of labels, but there is no encoder. " \
                                                        "The reason could be that the label was not added using " \
                                                        "add_label() or that it contains continuous values " \
                                                        "(LabelType.REGRESSION was specified when called add_label())."
        return self.__label_encoders[label]
