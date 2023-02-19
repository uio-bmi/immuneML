import logging
from datetime import datetime

import random
import numpy as np
import pkg_resources
import torch
from sklearn.preprocessing import label_binarize

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Constants import Constants


class Util:

    @staticmethod
    def map_to_old_class_values(y, class_mapping: dict):
        try:
            old_class_type = np.array(list(class_mapping.values())).dtype
            mapped_y = np.copy(y).astype(object)
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
            mapped_y = np.copy(y).astype(object)
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
    def make_class_mapping(y, positive_class=None) -> dict:
        """Creates a class mapping from a list of classes which can be strings, numbers of booleans; maps to same name in multi-class settings"""
        classes = np.unique(y)
        if classes.shape[0] == 2:
            return Util.make_binary_class_mapping(y, positive_class)
        else:
            return {cls: cls for cls in classes}

    @staticmethod
    def make_binary_class_mapping(y, positive_class=None) -> dict:
        """
        Creates binary class mapping from a list of classes which can be strings, numbers or boolean values

        Arguments:

            y: list of classes per example, as supplied to fit() method of the classifier; it should include all classes that will appear in the data

        Returns:
             mapping dictionary where 0 and 1 are always the keys and the values are original class names which were mapped for these values

        """
        unique_values = sorted(set(y))
        assert len(unique_values) == 2, f"MLMethod: there has two be exactly two classes to use this classifier," \
                                        f" instead got {str(unique_values)[1:-1]}. For multi-class classification, " \
                                        f"consider some of the other classifiers."

        if positive_class is None:
            return {0: unique_values[0], 1: unique_values[1]}
        else:
            assert positive_class in unique_values, f"MLMethod: the specified positive class '{positive_class}' does not occur " \
                                                    f"in the list of available classes: {str(unique_values)[1:-1]}."
            unique_values.remove(positive_class)
            return {0: unique_values[0], 1: positive_class}


    @staticmethod
    def binarize_label_classes(true_y, predicted_y, classes):
        """
        Binarizes the predictions in place using scikit-learn's label_binarize() method

        Necessary for some sklearn metrics, like roc_auc_score
        """
        if hasattr(true_y, 'dtype') and (true_y.dtype.type is np.str_ or true_y.dtype.type is np.object_) \
                or isinstance(true_y, list) and any(isinstance(item, str) for item in true_y):
            true_y = label_binarize(true_y, classes=classes)
            predicted_y = label_binarize(predicted_y, classes=classes)

        return true_y, predicted_y

    @staticmethod
    def setup_pytorch(number_of_threads, random_seed, pytorch_device_name=None):
        torch.set_num_threads(number_of_threads)
        torch.manual_seed(random_seed)
        if pytorch_device_name is not None:
            torch.device(pytorch_device_name)

    @staticmethod
    def get_immuneML_version():
        try:
            return 'immuneML ' + pkg_resources.get_distribution('immuneML').version
        except pkg_resources.DistributionNotFound as err:
            try:
                return 'immuneML ' + Constants.VERSION
            except Exception as e:
                return f'immuneML-dev-{datetime.now()}'

    @staticmethod
    def get_train_val_indices(n_examples, training_percentage, random_seed=None):
        indices = list(range(n_examples))

        random.seed(random_seed)
        random.shuffle(indices)
        random.seed(None)

        limit = int(n_examples * training_percentage)
        train_indices = indices[:limit]
        val_indices = indices[limit:]

        return train_indices, val_indices

    @staticmethod
    def subset_encoded_data(encoded_data: EncodedData, indices):
        return EncodedData(examples=encoded_data.examples[indices],
                           labels={label_name: [encoded_data.labels[label_name][i] for i in indices]
                                   for label_name in encoded_data.labels.keys()},
                           example_ids=[encoded_data.example_ids[i] for i in indices],
                           example_weights=[encoded_data.example_weights[i] for i in indices] if encoded_data.example_weights is not None else None,
                           feature_names=encoded_data.feature_names,
                           feature_annotations=encoded_data.feature_annotations,
                           encoding=encoded_data.encoding,
                           info=encoded_data.info)
