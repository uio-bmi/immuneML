from pathlib import Path

import numpy as np
from sonia.sequence_generation import SequenceGeneration
from sonnia.sonnia import SoNNia as InternalSoNNia

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import print_log


class SoNNia(GenerativeModel):
    """
    SoNNia models the selection process of T and B cell receptor repertoires. It is based on the SoNNia Python package.

    Original publication:
    Isacchini, G., Walczak, A. M., Mora, T., & Nourmohammad, A. (2021). Deep generative selection models of T and B
    cell receptor repertoires with soNNia. Proceedings of the National Academy of Sciences, 118(14), e2023141118.
    https://doi.org/10.1073/pnas.2023141118

    Arguments:

        chain (str)

        batch_size (int)

        epochs (int)

        deep (bool)

        include_joint_genes (bool)

        include_indep_genes (bool)

    """

    def __init__(self, chain=None, batch_size: int = None, epochs: int = None, deep: bool = False,
                 include_joint_genes: bool = True, include_indep_genes: bool = False):
        super().__init__(chain)
        self.epochs = epochs
        self.batch_size = batch_size
        self.deep = deep
        self.include_joint_genes = include_joint_genes
        self.include_indep_genes = include_indep_genes
        self._model = None

    def fit(self, data):
        # TODO: decide how to represent this data - what format should data be in, where should SoNNia-specific representation happen?
        print_log(f"{SoNNia.__name__}: fitting a selection model...", True)
        self._model = InternalSoNNia(data_seqs=data, deep=self.deep, include_joint_genes=self.include_joint_genes,
                                     include_indep_genes=self.include_indep_genes)
        self._model.infer_selection(self.epochs, self.batch_size)
        print_log(f"{SoNNia.__name__}: selection model fitted.", True)

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool):
        gen_model = SequenceGeneration(self._model)
        sequences = gen_model.generate_sequences_post(count)
        return sequences  # TODO: make some meaningful data structure here

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise NotImplementedError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise NotImplementedError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise NotImplementedError
