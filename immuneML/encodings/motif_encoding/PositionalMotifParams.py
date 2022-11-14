from dataclasses import dataclass

@dataclass
class PositionalMotifParams:
    max_positions: int
    count_threshold: int
    allow_negative_amino_acids: bool
    pool_size: int = 4