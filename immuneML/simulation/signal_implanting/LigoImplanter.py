import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.util import get_sequence_per_signal_count, make_sequences_from_gen_model, get_bnp_data, filter_out_illegal_sequences, \
    annotate_sequences
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class LigoImplanter:
    sim_item: LIgOSimulationItem = None
    sequence_type: SequenceType = None
    all_signals: List[Signal] = None
    sequence_batch_size: int = None
    seed: int = None
    export_p_gens: bool = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None
    max_iterations: int = None
    p_gen_threshold: float = None
    p_gen_histogram = None
    p_gen_bin_count: int = None
    keep_low_p_gen_proba: float = None

    @property
    def max_signals(self):
        return 0 if self.remove_seqs_with_signals else -1

    def make_repertoires(self, path: Path) -> List[Repertoire]:

        seqs_per_signal_count = get_sequence_per_signal_count(self.sim_item)
        iteration = 0

        while sum(seqs_per_signal_count.values()) > 0 and iteration < self.max_iterations:
            sequence_path = self._make_background_sequences(path)

            if self.keep_p_gen_dist and iteration == 0:
                self._make_p_gen_histogram(sequence_path)

            if self.remove_seqs_with_signals:
                self._filter_background_sequences(sequence_path)

            seqs_per_signal_count = self._implant_in_sequences()

            if self.keep_p_gen_dist or self.p_gen_threshold:
                seqs_per_signal_count = self._filter_using_p_gens()

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached during implanting.")
            iteration += 1

        repertoires = self._make_repertoire_objects()
        return repertoires

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path):
        raise NotImplementedError

    def _make_background_sequences(self, path) -> Path:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{self.seed}_{self.sim_item.name}.tsv"
        make_sequences_from_gen_model(self.sim_item, self.sequence_batch_size, self.seed, sequence_path, self.sequence_type, False)
        return sequence_path

    def _filter_background_sequences(self, sequence_path: Path):
        background_sequences = get_bnp_data(sequence_path)
        annotated_sequences = annotate_sequences(background_sequences, self.sequence_type == SequenceType.AMINO_ACID, self.all_signals)
        if self.remove_seqs_with_signals:
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self.sim_item, self.all_signals, self.max_signals)
        return annotated_sequences

    def _implant_in_sequences(self):
        raise NotImplementedError

    def _make_p_gen_histogram(self, sequence_path):
        if not self.sim_item.generative_model.can_compute_p_gens():
            raise RuntimeError(f"{LigoImplanter.__name__}: generative model of class {type(self.sim_item.generative_model).__name__} cannot "
                               f"compute sequence generation probabilities. Use other generative model or set keep_p_gen_dist parameter to False.")

        sequence_df = pd.read_csv(sequence_path, sep='\t')
        p_gens = self.sim_item.generative_model.compute_p_gens(sequence_df, self.sequence_type)
        self.p_gen_histogram = np.histogram(np.log10(p_gens), bins=self.p_gen_bin_count, density=True)

    def _distribution_matches(self) -> bool:
        raise NotImplementedError

    def _filter_using_p_gens(self):
        raise NotImplementedError

    def _make_repertoire_objects(self):
        raise NotImplementedError
