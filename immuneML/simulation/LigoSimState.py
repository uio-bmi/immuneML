from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.simulation.SimConfig import SimConfig


@dataclass
class LigoSimState:
    signals: list
    simulation: SimConfig
    paths: dict = None
    name: str = None
    formats = None

    # defined at runtime
    target_p_gen_histogram: Dict[str, np.ndarray] = field(default_factory=dict)
    p_gen_bins: Dict[str, Any] = field(default_factory=dict)
    resulting_dataset: Dataset = None
    result_path: Path = None
