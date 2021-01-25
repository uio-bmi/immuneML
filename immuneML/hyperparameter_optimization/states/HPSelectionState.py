from pathlib import Path

from immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class HPSelectionState:

    def __init__(self, train_datasets, val_datasets, path: Path, hp_strategy: HPOptimizationStrategy):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.path = path
        self.hp_strategy = hp_strategy.clone()
        self.hp_items = {str(hp_setting): [] for hp_setting in self.hp_strategy.hp_settings}
        self.train_data_reports = []
        self.val_data_reports = []
        self.data_reports = []

    @property
    def optimal_hp_setting(self):
        return self.hp_strategy.get_optimal_hps()
