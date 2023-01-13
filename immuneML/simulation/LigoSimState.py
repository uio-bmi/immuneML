from dataclasses import dataclass
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.simulation.SimConfig import SimConfig


@dataclass
class LigoSimState:
    signals: list
    simulation: SimConfig
    store_signal_in_receptors: bool
    paths: dict = None
    name: str = None
    formats = None

    # defined at runtime
    target_p_gen_histogram: np.ndarray = None
    p_gen_bins = None
    resulting_dataset: Dataset = None
    result_path: Path = None
