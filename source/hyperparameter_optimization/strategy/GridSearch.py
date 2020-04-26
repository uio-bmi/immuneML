from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.HPSettingResult import HPSettingResult
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class GridSearch(HPOptimizationStrategy):

    def generate_next_setting(self, hp_setting: HPSetting = None, metric: float = None) -> HPSetting:

        if hp_setting is not None:
            self.search_space_metric[hp_setting.get_key()] = metric

        keys = [key for key in self.search_space_metric if self.search_space_metric[key] is None]

        if len(keys) > 0:
            next_setting = self.hp_settings[keys[0]]
        else:
            next_setting = None

        return next_setting

    def get_optimal_hps(self) -> HPSetting:
        optimal_key = max(self.search_space_metric, key=lambda k: self.search_space_metric[k])
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
        return GridSearch(hp_settings=self.hp_settings.values())
