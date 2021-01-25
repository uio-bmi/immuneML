import abc

from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.HPSettingResult import HPSettingResult


class HPOptimizationStrategy(metaclass=abc.ABCMeta):
    """
    hyper-parameter optimization strategy is a base class of all different hyper-parameter optimization approaches,
    such as grid search, random search, bayesian optimization etc.

    HPOptimizationStrategy internally keeps a dict of settings that were tried out and the metric value that was
    obtained on the validation set which it then uses to determine the next step
    """

    def __init__(self, hp_settings: list, search_criterion=max):
        self.hp_settings = {hp_setting.get_key(): hp_setting for hp_setting in hp_settings}
        self.search_space_metric = {hp_setting.get_key(): None for hp_setting in hp_settings}
        self.search_criterion = search_criterion

    @abc.abstractmethod
    def generate_next_setting(self, hp_setting: HPSetting = None, metric: dict = None):
        """
        generator function which returns the next hyper-parameter setting to be evaluated
        :param hp_setting: previous setting (None if it's the first iteration)
        :param metric: performance metric from the previous setting per label
        :return: new hp_setting or None (if the end is reached)
        """
        pass

    @abc.abstractmethod
    def get_optimal_hps(self) -> HPSetting:
        pass

    @abc.abstractmethod
    def get_all_hps(self) -> HPSettingResult:
        pass

    @abc.abstractmethod
    def clone(self):
        pass
