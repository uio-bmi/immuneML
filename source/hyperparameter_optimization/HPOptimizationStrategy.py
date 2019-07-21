import abc

from source.hyperparameter_optimization.HPSetting import HPSetting


class HPOptimizationStrategy(metaclass=abc.ABCMeta):
    """
    hyper-parameter optimization strategy is a base class of all different hyper-parameter optimization approaches,
    such as grid search, random search, bayesian optimization etc.

    HPOptimizationStrategy internally keeps a dict of settings that were tried out and the metric value that was
    obtained on the validation set which it then uses to determine the next step
    """

    def __init__(self, encoders: list, ml_methods: list, preprocessing_sequences: list):
        self.encoders = encoders
        self.ml_methods = ml_methods
        self.preprocessing_sequences = preprocessing_sequences

        self.search_space_metric = {}
        for e in range(len(self.encoders)):
            for ml in range(len(self.ml_methods)):
                for p in range(len(self.preprocessing_sequences)):
                    self.search_space_metric[self.build_key(e, ml, p)] = -1

    def build_key(self, encoding_index, ml_method_index, preproc_sequence_index):
        return "enc{}_ml{}_preproc{}".format(encoding_index, ml_method_index, preproc_sequence_index)

    @abc.abstractmethod
    def get_next_setting(self, hp_setting: HPSetting = None, metric: float = -1):
        """
        generator function which returns the next hyper-parameter setting to be evaluated
        :param hp_setting: previous setting (None if it's the first iteration)
        :param metric: performance metric from the previous setting (-1 if it is the first iteration)
        :return: yields new hp_setting or None (if the end is reached)
        """
        pass

    @abc.abstractmethod
    def get_optimal_hps(self) -> HPSetting:
        pass
