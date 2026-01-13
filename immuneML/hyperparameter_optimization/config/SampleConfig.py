from dataclasses import dataclass


@dataclass
class SampleConfig:
    """Configuration for dataset splitting into subsets without replacement with given number of repetitions and
    given percentage of the original dataset."""
    percentage: float = 0.8
    split_count: int = 5
    random_seed: int = None