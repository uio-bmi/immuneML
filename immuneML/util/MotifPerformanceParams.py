from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.RegionType import RegionType


@dataclass
class MotifPerformanceParams:
    result_path: Path = None
    training_set_name: str = None
    test_set_name: str = None
    determine_tp_cutoff: bool = None
    split_by_motif_size: bool = None
    highlight_motifs_path: str = None
    highlight_motifs_name: str = None
    min_points_in_window: int = None
    smoothing_constant1: float = None
    smoothing_constant2: float = None
    test_precision_threshold: float = None
    n_positives_in_training_data: int = None
    dataset_size: int = None
    class_name: str = None
    report_name: str = None

