from datetime import datetime
from typing import List

import numpy as np
import pkg_resources
import torch


class Util:

    @staticmethod
    def make_binary_class_mapping(y, label_names: List[str]) -> dict:
        """
        Creates binary class mapping from a list of classes which can be strings, numbers or boolean values

        Arguments:

            y: list of classes per example, as supplied to fit() method of the classifier; it should include all classes that will appear in the data

            label_names: the list of names of labels for which the classes are supplied (for binary classification, there should be only one label)

        Returns:
             mapping dictionary where 0 and 1 are always the keys and the values are original class names which were mapped for these values

        """
        assert len(label_names) == 1, "MLMethod: more than one label was specified at the time, but this " \
                                      "classifier can handle only binary classification for one label. Try using TrainMLModel " \
                                      "instruction which will train different classifiers for all provided labels."
        unique_values = np.sort(np.unique(y[label_names[0]]))
        assert unique_values.shape[0] == 2, f"MLMethod: there has two be exactly two classes to use this classifier," \
                                            f" instead got {str(unique_values.tolist())[1:-1]}. For multi-class classification, " \
                                            f"consider some of the other classifiers."

        if 0 in unique_values and 1 in unique_values and unique_values.dtype != bool:
            mapping = {0: 0, 1: 1}
        elif True in unique_values and False in unique_values:
            mapping = {0: False, 1: True}
        else:
            mapping = {0: unique_values[0], 1: unique_values[1]}

        return mapping

    @staticmethod
    def setup_pytorch(number_of_threads, random_seed):
        torch.set_num_threads(number_of_threads)
        torch.manual_seed(random_seed)

    @staticmethod
    def get_immuneML_version():
        try:
            return 'immuneML ' + pkg_resources.get_distribution('immuneML').version
        except pkg_resources.DistributionNotFound as err:
            return f'immuneML-dev-{datetime.now()}'
