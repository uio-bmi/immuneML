import copy

from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.HPSettingResult import HPSettingResult
from immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class GridSearch(HPOptimizationStrategy):

    def generate_next_setting(self, hp_setting: HPSetting = None, metric: float = None) -> HPSetting:

        if hp_setting is not None:
            self.search_space_metric[hp_setting.get_key()] = metric

        keys = [key for key in self.search_space_metric if self.search_space_metric[key] is None]

        if len(keys) > 0:
            next_setting = self.hp_settings[keys[0]]
        else:
            next_setting = None

        return copy.deepcopy(next_setting)

    def get_optimal_hps(self) -> HPSetting:
        """
        Finds the optimal hyperparameter setting, where the optimal is the one with max/min value of the search metric.
        The search criterion (object attribute) defines if it should be max (its value is max function) or min (its value is min
        function). max corresponds to metrics such as accuracy, AUC, while min corresponds to metrics such as log loss.

        Returns:
            HPSetting object which had the optimal performance based on the metric value in the search space

        """
        if len(list(self.search_space_metric.keys())) == 1:
            optimal_key = list(self.search_space_metric.keys())[0]
        else:
            optimal_key = self.search_criterion({key: value for key, value in self.search_space_metric.items() if isinstance(value, float)},
                                                key=lambda k: self.search_space_metric[k])
        return self.hp_settings[optimal_key]

    def get_all_hps(self) -> HPSettingResult:
        optimal_setting = self.get_optimal_hps()
        res = HPSettingResult(optimal_setting=optimal_setting, all_settings=self.hp_settings)
        return res

    def get_performance(self, hp_setting: HPSetting):
        key = hp_setting.get_key()
        if key in self.search_space_metric:
            return self.search_space_metric[key]
        else:
            return None

    def clone(self):
        return GridSearch(hp_settings=self.hp_settings.values(), search_criterion=self.search_criterion)
