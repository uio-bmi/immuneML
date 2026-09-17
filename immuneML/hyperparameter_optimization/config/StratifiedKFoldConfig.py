from dataclasses import dataclass


@dataclass
class StratifiedKFoldConfig:
    stratification_label: str = None
