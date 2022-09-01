from dataclasses import dataclass

@dataclass
class PositionalMotifParams:
    max_positions: int
    count_threshold: int
    pool_size: int = 4
