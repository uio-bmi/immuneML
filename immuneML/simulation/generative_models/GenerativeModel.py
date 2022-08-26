import abc
from pathlib import Path

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
    def compute_p_gens(self):
        pass

    @abc.abstractmethod
    def can_compute_p_gens(self) -> bool:
        pass

    @abc.abstractmethod
    def can_generate_from_skewed_gene_models(self) -> bool:
        pass

    @abc.abstractmethod
    def generate_from_skewed_gene_models(self, gene_pair_probabilities: dict) -> bool:
        pass
