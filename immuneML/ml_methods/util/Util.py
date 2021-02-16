import logging
from datetime import datetime

import numpy as np
import pkg_resources
import torch
from sklearn.preprocessing import label_binarize

from immuneML.environment.Constants import Constants


class Util:

    @staticmethod
    def map_to_old_class_values(y, class_mapping: dict):
        try:
            old_class_type = np.array(list(class_mapping.values())).dtype
            mapped_y = np.copy(y).astype(np.object)
            for i in range(mapped_y.shape[0]):
                mapped_y[i] = class_mapping[y[i]]
            return mapped_y.astype(old_class_type)
        except Exception as e:
            logging.exception("MLMethod util: error occurred when predicting the class assignment due to mismatch of class types.\n"
                              f"Classes: {y}\nMapping:{class_mapping}")
            raise e

    @staticmethod
    def map_to_new_class_values(y, class_mapping: dict):
        try:
            mapped_y = np.copy(y).astype(np.object)
            switched_mapping = {value: key for key, value in class_mapping.items()}
            new_class_type = np.array(list(switched_mapping.values())).dtype
            for i in range(mapped_y.shape[0]):
                mapped_y[i] = switched_mapping[y[i]]
            return mapped_y.astype(new_class_type)
        except Exception as e:
            logging.exception(f"MLMethod util: error occurred when fitting the model due to mismatch of class types.\n"
                              f"Classes: {y}\nMapping:{class_mapping}")
            raise e

    @staticmethod
    def make_class_mapping(y) -> dict:
        """Creates a class mapping from a list of classes which can be strings, numbers of booleans; maps to same name in multi-class settings"""
        classes = np.unique(y)
        if classes.shape[0] == 2:
            return Util.make_binary_class_mapping(y)
        else:
            return {cls: cls for cls in classes}

    @staticmethod
    def make_binary_class_mapping(y) -> dict:
        """
        Creates binary class mapping from a list of classes which can be strings, numbers or boolean values

        Arguments:

            y: list of classes per example, as supplied to fit() method of the classifier; it should include all classes that will appear in the data

        Returns:
             mapping dictionary where 0 and 1 are always the keys and the values are original class names which were mapped for these values

        """
        unique_values = np.sort(np.unique(y))
        assert unique_values.shape[0] == 2, f"MLMethod: there has two be exactly two classes to use this classifier," \
                                            f" instead got {str(unique_values.tolist())[1:-1]}. For multi-class classification, " \
                                            f"consider some of the other classifiers."

        if 0 == unique_values[0] and 1 == unique_values[1] and unique_values.dtype != bool:
            mapping = {0: 0, 1: 1}
        elif 0 == unique_values[0] and 1 == unique_values[1] and unique_values.dtype == bool:
            mapping = {0: False, 1: True}
        else:
            mapping = {0: unique_values[0], 1: unique_values[1]}

        return mapping

    @staticmethod
    def binarize_labels(true_y, predicted_y, labels):
        """Binarizes the predictions in place using scikit-learn's label_binarize() method"""
        if hasattr(true_y, 'dtype') and true_y.dtype.type is np.str_ or isinstance(true_y, list) and any(isinstance(item, str) for item in true_y):
            true_y = label_binarize(true_y, classes=labels)
            predicted_y = label_binarize(predicted_y, classes=labels)

        return true_y, predicted_y

    @staticmethod
    def setup_pytorch(number_of_threads, random_seed):
        torch.set_num_threads(number_of_threads)
        torch.manual_seed(random_seed)

    @staticmethod
    def get_immuneML_version():
        try:
            return 'immuneML ' + pkg_resources.get_distribution('immuneML').version
        except pkg_resources.DistributionNotFound as err:
            try:
                return 'immuneML ' + Constants.VERSION
            except Exception as e:
                return f'immuneML-dev-{datetime.now()}'
