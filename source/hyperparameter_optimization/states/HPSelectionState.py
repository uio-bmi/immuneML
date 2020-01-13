from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy


class HPSelectionState:

    def __init__(self, train_datasets, val_datasets, path: str, hp_strategy: HPOptimizationStrategy):
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.path = path
        self.hp_strategy = hp_strategy.clone()

    @property
    def optimal_hp_setting(self):
        return self.hp_strategy.get_optimal_hps()
