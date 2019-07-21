from source.hyperparameter_optimization.SplitType import SplitType


class SplitConfig:

    def __init__(self, split_strategy: SplitType, split_count: int,
                 training_percentage: float = None, label_to_balance: str = None):
        self.split_strategy = split_strategy
        self.split_count = split_count
        self.training_percentage = training_percentage
        self.label_to_balance = label_to_balance
