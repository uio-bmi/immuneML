from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ComparisonDataBatch:
    """
    matrix: array with dimension items x repertoires, where items are defined by comparison attributes specified in ComparisonData
            class and can include, for instance, receptor sequences or combinations of receptor sequences and V and J gene
    items: the item names extracted from the repertoires in the dataset on which the repertoires are evaluated (e.g. sequences or
            combinations of sequences and genes
    repertoire_index_mapping: a mapping between the repertoire identifier (a string) and a column number for faster access of columns
            (repertoire vectors w.r.t. given items) in the comparison data matrix where columns correspond to repertoires
    """

    matrix: np.ndarray
    items: list
    repertoire_index_mapping: Dict[str, int]
