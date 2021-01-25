from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManualSplitConfig:

    train_metadata_path: Path = None
    test_metadata_path: Path = None
