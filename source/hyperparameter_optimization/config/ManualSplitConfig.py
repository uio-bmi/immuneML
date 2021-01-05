from pathlib import Path
from dataclasses import dataclass


@dataclass
class ManualSplitConfig:

    train_metadata_path: Path = None
    test_metadata_path: Path = None
