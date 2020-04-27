from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitType import SplitType


class SplitConfig:

    def __init__(self, split_strategy: SplitType, split_count: int, training_percentage: float = None, reports: ReportConfig = None):
        self.split_strategy = split_strategy
        self.split_count = split_count
        self.training_percentage = training_percentage
        self.reports = reports if reports is not None else ReportConfig()

    def __str__(self):
        desc = ""
        if self.split_strategy == SplitType.K_FOLD:
            desc = f"{self.split_count}-fold CV"
        elif self.split_strategy == SplitType.RANDOM:
            desc = f"{self.split_count}-fold MC CV (training percentage: {self.training_percentage})"
        elif self.split_strategy == SplitType.LOOCV:
            desc = "LOOCV"
        return desc
