from datetime import datetime

import numpy as np
import pkg_resources
import torch

from immuneML.environment.Constants import Constants


class Util:

    @staticmethod
    def make_binary_class_mapping(y, label_name: str) -> dict:
        """
        Creates binary class mapping from a list of classes which can be strings, numbers or boolean values

        Arguments:

            y: list of classes per example, as supplied to fit() method of the classifier; it should include all classes that will appear in the data

            label_name: the name of the label for which the classes are supplied

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
