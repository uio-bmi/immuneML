from dataclasses import dataclass
from pathlib import Path

@dataclass
class CompAIRRParams:
    compairr_path: Path
    keep_compairr_input: bool
    differences: int
    indels: bool
    ignore_counts: bool
    ignore_genes: bool
    threads: int
    output_filename: str
    log_filename: str
    output_pairs: bool
    pairs_filename: str
    is_cdr3: bool = False
    do_repertoire_overlap: bool = True
    do_sequence_matching: bool = False
