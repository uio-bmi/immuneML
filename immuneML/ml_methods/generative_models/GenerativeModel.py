import abc
from pathlib import Path

import numpy as np
import logging

from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType


class GenerativeModel:
    """

    Generative models are algorithms which can be trained to learn patterns in existing datasets,
    and then be used to generate new synthetic datasets.

    These methods can be used in the :ref:`TrainGenModel` instruction, and previously trained
    models can be used to generate data using the :ref:`ApplyGenModel` instruction.
    """

    DOCS_TITLE = "Generative models"
    OUTPUT_COLUMNS = []

    def __init__(self, locus: Chain, name: str = None, region_type: RegionType = None, seed=None):
        self.locus = Chain.get_chain(locus) if locus is not None else None
        self.name = name
        self.region_type = region_type
        self.seed = seed

    @abc.abstractmethod
    def fit(self, data, path: Path = None):
        pass

    @abc.abstractmethod
    def is_same(self, model) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool) \
            -> Dataset:
        pass

    @abc.abstractmethod
    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        pass

    @abc.abstractmethod
    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        pass

    @abc.abstractmethod
    def can_compute_p_gens(self) -> bool:
        pass

    @abc.abstractmethod
    def can_generate_from_skewed_gene_models(self) -> bool:
        pass

    @abc.abstractmethod
    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int,
                                         compute_p_gen: bool):
        pass

    @abc.abstractmethod
    def save_model(self, path: Path) -> Path:
        pass

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path: Path):
        pass

    def set_locus(self, dataset: Dataset):
        dataset_locus = dataset.get_locus()
        if len(dataset_locus) > 0:
            logging.info(f"GenerativeModel: input dataset has multiple loci, choosing: {dataset_locus}")

        if self.locus is not None and dataset_locus[0] != self.locus.value:
            logging.info(f"GenerativeModel: Overwriting default locus {self.locus.value} with dataset locus {dataset_locus[0]}")

        self.locus = Chain.get_chain(dataset_locus[0])
