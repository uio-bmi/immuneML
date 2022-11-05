import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.MotifInstance import MotifInstance
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
            sequences = self._make_background_sequences(path)

            if self.keep_p_gen_dist and iteration == 0:
                self._make_p_gen_histogram(sequences)

            if self.remove_seqs_with_signals:
                sequences = self._filter_background_sequences(sequences)

            sequences, seqs_per_signal_count = self._implant_in_sequences(sequences, seqs_per_signal_count)

            if self.keep_p_gen_dist or self.p_gen_threshold:
                seqs_per_signal_count = self._filter_using_p_gens(sequences, seqs_per_signal_count)

            if iteration == int(self.max_iterations * 0.75):
                logging.warning(f"Iteration {iteration} out of {self.max_iterations} max iterations reached during implanting.")
            iteration += 1

        if iteration == self.max_iterations and sum(seqs_per_signal_count.values()) != 0:
            raise RuntimeError(f"{LigoImplanter.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self)}.\n")

        repertoires = self._make_repertoire_objects()
        return repertoires

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path):
        raise NotImplementedError

    def _make_background_sequences(self, path) -> bnpdataclass:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{self.seed}_{self.sim_item.name}.tsv"
        make_sequences_from_gen_model(self.sim_item, self.sequence_batch_size, self.seed, sequence_path, self.sequence_type, False)
        return get_bnp_data(sequence_path)

    def _filter_background_sequences(self, sequences: bnpdataclass) -> bnpdataclass:
        annotated_sequences = annotate_sequences(sequences, self.sequence_type == SequenceType.AMINO_ACID, self.all_signals)
        if self.remove_seqs_with_signals:
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self.sim_item, self.all_signals, self.max_signals)
        return annotated_sequences

    def _implant_in_sequences(self, sequences: bnpdataclass, seqs_per_signal_count: dict) -> bnpdataclass:

        sequence_lengths = getattr(sequences, 'sequence_aa' if self.sequence_type == SequenceType.AMINO_ACID else 'sequence').shape.lengths
        sorted_indices = np.argsort(sequence_lengths)

        motif_instances = self._make_motif_instances(seqs_per_signal_count)

        for signal_id, instances in motif_instances.items():
            instance_lengths = np.array([len(instance) for instance in instances])
            motif_sorted_indices = np.argsort(instance_lengths)
            min_len = min(instance_lengths.shape[0], sequence_lengths.shape[0])

            motif_fit_mask = sequence_lengths[sorted_indices][:min_len] > instance_lengths[motif_sorted_indices][:min_len]
            selected_indices = sorted_indices[:min_len][motif_fit_mask]


    def _make_motif_instances(self, seqs_per_signal_count: dict):
        instances = {signal.id: signal.make_motif_instances(seqs_per_signal_count[signal.id], self.sequence_type) for signal in self.sim_item.signals
                     if seqs_per_signal_count[signal.id] > 0}

        if any(any(not isinstance(el, MotifInstance) for el in signal_motifs) for signal_id, signal_motifs in instances.items()):
            raise NotImplementedError("When using implanting, V and J genes must not been set in the motifs -- V/J gene implanting is not supported.")

        return instances

    def _make_p_gen_histogram(self, sequences: bnpdataclass):
        if not self.sim_item.generative_model.can_compute_p_gens():
            raise RuntimeError(f"{LigoImplanter.__name__}: generative model of class {type(self.sim_item.generative_model).__name__} cannot "
                               f"compute sequence generation probabilities. Use other generative model or set keep_p_gen_dist parameter to False.")

        p_gens = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)
        self.p_gen_histogram = np.histogram(np.log10(p_gens), bins=self.p_gen_bin_count, density=True)

    def _distribution_matches(self) -> bool:
        raise NotImplementedError

    def _filter_using_p_gens(self, sequences: bnpdataclass, seqs_per_signal_count: dict):
        raise NotImplementedError

    def _make_repertoire_objects(self):
        raise NotImplementedError
