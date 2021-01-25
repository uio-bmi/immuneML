from immuneML.hyperparameter_optimization.HPSetting import HPSetting


class HPSettingResult:
    """
    HPSettingResult encapsulates the results from evaluating a set of different hyperparameter settings (e.g. on one train/test split
    in the outer loop of nested cross-validation) - it stores the optimal setting which can be used to assess the performance on the task,
    and all settings if needed for downstream analysis.
    """
    def __init__(self, optimal_setting: HPSetting, all_settings: dict):
        self.optimal_setting = optimal_setting

        assert all(isinstance(setting, HPSetting) for setting in all_settings.values())
        self.all_settings = all_settings
