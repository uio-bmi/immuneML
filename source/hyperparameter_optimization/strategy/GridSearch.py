from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class GridSearch(HPOptimizationStrategy):

    def get_next_setting(self, hp_setting: HPSetting = None, metric_per_label: dict = None) -> HPSetting:

        if hp_setting is not None:
            self.search_space_metric[hp_setting.get_key()] = metric_per_label

        keys = [key for key in self.search_space_metric if self.search_space_metric[key] is None]

        if len(keys) > 0:
            next_setting = self.hp_settings[keys[0]]
        else:
            next_setting = None

        return next_setting

    def get_optimal_hps(self) -> HPSetting:
        """
        :return: the hyper-parameter setting which has the highest average performance across all labels
        """
        print(self.search_space_metric)
        key = max(self.search_space_metric.keys(), key=(lambda k: sum(self.search_space_metric[k][label]
                                                                      for label in self.search_space_metric[k])
                                                                  / len(self.search_space_metric[k])))
        return self.hp_settings[key]
