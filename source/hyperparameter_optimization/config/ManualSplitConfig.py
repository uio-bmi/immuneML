from dataclasses import dataclass


@dataclass
class ManualSplitConfig:

    train_metadata_path: str = None
    test_metadata_path: str = None
