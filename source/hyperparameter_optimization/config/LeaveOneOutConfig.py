from dataclasses import dataclass


@dataclass
class LeaveOneOutConfig:

    parameter: str = None
    min_count: int = None
