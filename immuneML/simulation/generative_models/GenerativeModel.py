import abc
from pathlib import Path

import numpy as np

from immuneML.environment.SequenceType import SequenceType


class GenerativeModel:

    OUTPUT_COLUMNS = []

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def generate_sequences(self, count: int, seed: int = None, path: Path = None, sequence_type: SequenceType = SequenceType.AMINO_ACID):
        pass

    @abc.abstractmethod
    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        pass

    @abc.abstractmethod
    def can_compute_p_gens(self) -> bool:
        pass

    @abc.abstractmethod
    def can_generate_from_skewed_gene_models(self) -> bool:
        pass

    @abc.abstractmethod
    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path, sequence_type: SequenceType, batch_size: int):
        pass
