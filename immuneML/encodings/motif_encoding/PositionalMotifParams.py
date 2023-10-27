from dataclasses import dataclass

@dataclass
class PositionalMotifParams:
    max_positions: int
    min_positions: int
    count_threshold: int
    pool_size: int = 4
    allow_negative_aas: bool = False
