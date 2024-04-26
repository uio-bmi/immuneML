import copy
import os
import random
from dataclasses import fields, field
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Tuple

import math

import dill
import numpy as np
from bionumpy.bnpdataclass import BNPDataClass

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.app.LigoApp import SimError
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.LigoSimState import LigoSimState
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.ImplantingStrategy import ImplantingStrategy
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.simulation.util.util import get_bnp_data, make_receptor_sequence_objects, make_annotated_dataclass, \
    get_sequence_per_signal_count, \
    update_seqs_without_signal, update_seqs_with_signal, check_iteration_progress, make_sequence_paths, \
    make_signal_metadata, needs_seqs_with_signal, \
    check_sequence_count, make_repertoire_from_sequences, get_no_signal_sequences, get_signal_sequences, \
    annotate_sequences, get_signal_sequence_count, filter_sequences_by_length
from immuneML.util.ExporterHelper import ExporterHelper
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ligo_simulation.runtime_reports import make_p_gen_histogram_plot


class LigoSimInstruction(Instruction):
    """
    LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
    the user.

    **Specification arguments:**

    - simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

    - sequence_batch_size (int): how many sequences to generate at once using the generative model before checking for signals and filtering

    - max_iterations (int): how many iterations are allowed when creating sequences

    - export_p_gens (bool): whether to compute generation probabilities (if supported by the generative model) for sequences and include them as part of output

    - number_of_processes (int): determines how many simulation items can be simulated in parallel


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        instructions:
            my_simulation_instruction: # user-defined name of the instruction
                type: LIgOSim # which instruction to execute
                simulation: sim1
                sequence_batch_size: 1000
                max_iterations: 1000
                export_p_gens: False
                number_of_processes: 4

    """

    def __init__(self, simulation: SimConfig, signals: List[Signal], name: str,
                 sequence_batch_size: int, max_iterations: int, number_of_processes: int, export_p_gens: bool = None):

        self.state = LigoSimState(simulation=simulation, signals=signals, name=name)
        self._number_of_processes = number_of_processes
        self._sequence_batch_size = sequence_batch_size
        self._max_iterations = max_iterations
        self._export_p_gens = export_p_gens

        self._use_p_gens = self.state.simulation.keep_p_gen_dist and \
                           all(sim_item.generative_model.can_compute_p_gens() for sim_item in
                               self.state.simulation.sim_items)

        self._export_observed_signals = any(
            [it.false_negative_prob_in_receptors > 0 or it.false_positive_prob_in_receptors > 0
             for it in self.state.simulation.sim_items])

        self._noise_fields = [(f"observed_{s.id}", int) for s in
                              self.state.signals] if self._export_observed_signals else []

        if isinstance(self.state.simulation.simulation_strategy, ImplantingStrategy):
            field_seq, field_p_gen = field(default='', metadata=None), field(default=-1., metadata=None)
            implanting_fields = [('original_sequence', str, dill.dumps(field_seq)),
                                 ('original_p_gen', float, dill.dumps(field_p_gen))]
        else:
            implanting_fields = []

        self._annotation_fields = sorted(
            [(signal.id, int) for signal in self.state.signals] + [('signals_aggregated', str)] +
            [(f"{signal.id}_positions", str) for signal in self.state.signals] + self._noise_fields,
            key=lambda x: x[0]) + implanting_fields

        self._custom_fields = self._annotation_fields + [('p_gen', float), ('from_default_model', int)]
        self._background_fields = [(f.name, f.type) for f in fields(BackgroundSequences)]

    @property
    def sequence_type(self) -> SequenceType:
        return self.state.simulation.sequence_type

    @property
    def _annotated_dataclass(self):
        return make_annotated_dataclass(self._annotation_fields, self.state.signals)

    MIN_RANGE_PROBABILITY = 1e-5

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        self._simulate_dataset()
        self._export_dataset()

        return self.state

    def _export_dataset(self):

        exporter_output = ExporterHelper.export_dataset(self.state.resulting_dataset, [AIRRExporter],
                                                        self.state.result_path, omit_columns=['from_default_model'])

        self.state.formats = exporter_output['formats']
        self.state.paths = exporter_output['paths']

    def _simulate_dataset(self):

        examples = self._create_examples_wrapper()
        random.shuffle(examples)

        labels = {**{signal.id: [True, False] for signal in self.state.signals},
                  **{'species': self.state.simulation.species}}

        if self.state.simulation.is_repertoire:
            self.state.resulting_dataset = RepertoireDataset.build_from_objects(labels=labels, repertoires=examples,
                                                                                name='simulated_dataset',
                                                                                metadata_path=self.state.result_path / 'metadata.csv',
                                                                                path=self.state.result_path)
        elif self.state.simulation.paired:
            self.state.resulting_dataset = ReceptorDataset.build_from_objects(examples, path=self.state.result_path,
                                                                              name='simulated_dataset',
                                                                              file_size=SequenceDataset.DEFAULT_FILE_SIZE,
                                                                              labels=labels)
        else:
            self.state.resulting_dataset = SequenceDataset.build_from_objects(examples, path=self.state.result_path,
                                                                              name='simulated_dataset',
                                                                              file_size=SequenceDataset.DEFAULT_FILE_SIZE,
                                                                              labels=labels)

    def _create_examples_wrapper(self) -> list:
        if self._number_of_processes > 1:
            chunk_size = math.ceil(len(self.state.simulation.sim_items) / self._number_of_processes)

            with Pool(processes=max(self._number_of_processes, len(self.state.simulation.sim_items))) as pool:
                result = pool.map(self._create_examples, [dill.dumps(item) for item in self.state.simulation.sim_items],
                                  chunksize=chunk_size)
                examples = {k: v for d in result for k, v in d.items()}
        else:
            examples = {}
            for item in self.state.simulation.sim_items:
                examples = {**examples, **self._create_examples(item)}

        if self.state.simulation.paired:
            examples = self._pair_examples(examples, self.state.result_path / 'paired')
        else:
            examples = list(chain.from_iterable(examples.values()))

        return examples

    def _pair_examples(self, examples: Dict[str, list], path: Path) -> list:
        paired_examples = []
        pair_func = self._pair_repertoires if self.state.simulation.is_repertoire else self._pair_sequences
        for paired_item1, paired_item2 in self.state.simulation.paired:
            paired_examples.extend(pair_func(examples[paired_item1], examples[paired_item2], path))
        return paired_examples

    def _pair_repertoires(self, repertoires1: list, repertoires2: list, path: Path) -> List[Repertoire]:
        assert len(repertoires1) == len(
            repertoires2), f"{LigoSimInstruction.__name__}: cannot create paired repertoires, number of repertoires per chain don't match: {len(repertoires1)} and {len(repertoires2)}."
        PathBuilder.build(path)
        paired_repertoires = []
        for i in range(len(repertoires1)):
            repertoire = self._pair_two_repertories(repertoires1[i], repertoires2[i], path)
            paired_repertoires.append(repertoire)
        return paired_repertoires

    def _pair_two_repertories(self, repertoire1: Repertoire, repertoire2: Repertoire, path: Path) -> Repertoire:
        assert repertoire1.get_element_count() == repertoire2.get_element_count(), f"{LigoSimInstruction.__name__}: cannot pair repertoires {repertoire1.identifier} and {repertoire2.identifier}, they have different number of sequences: {repertoire1.get_element_count()} and {repertoire2.get_element_count()}."

        sequences = []
        sequences1, sequences2 = repertoire1.sequences, repertoire2.sequences
        for index, seq1, seq2 in zip(list(range(len(sequences1))), sequences1, sequences2):
            seq1.metadata.cell_id = index
            seq2.metadata.cell_id = index
            seq1.sequence_id = f"{seq1.sequence_id}_{seq1.metadata.chain.value}"
            seq2.sequence_id = f"{seq2.sequence_id}_{seq2.metadata.chain.value}"
            sequences.extend([seq1, seq2])

        return Repertoire.build_from_sequence_objects(sequences, path,
                                                      metadata={**repertoire1.metadata, **repertoire2.metadata})

    def _pair_sequences(self, sequences1: list, sequences2: list, path: Path = None) -> List[Receptor]:
        assert len(sequences1) == len(sequences2), (f"{LigoSimInstruction.__name__}: could not create paired dataset, "
                                                    f"the number of sequences in two simulation items did not match.")

        random.shuffle(sequences1)
        random.shuffle(sequences2)

        return ReceptorBuilder.build_objects_from_pairs(sequences1, sequences2)

    def _create_examples(self, item_in) -> Dict[str, list]:

        item = dill.loads(item_in) if isinstance(item_in, bytes) else item_in

        if self.state.simulation.is_repertoire:
            res = self._create_repertoires(item)
        else:
            res = self._create_receptors(item)

        return {item.name: res}

    def _create_receptors(self, sim_item: SimConfigItem) -> list:

        assert len(sim_item.signals) in [0,
                                         1], f"{LigoSimInstruction.__name__}: for sequence datasets, only 0 or 1 signal or a signal pair per " \
                                             f"sequence are supported, but {len(sim_item.signals)} were specified."

        sequence_paths = self._gen_necessary_sequences(self.state.result_path, sim_item)
        sequences = None

        if len(sim_item.signals) == 0:
            sequences = get_bnp_data(sequence_paths['no_signal'], self._annotated_dataclass)
        else:
            for signal in sim_item.signals:
                signal_sequences = get_bnp_data(sequence_paths[signal.id], self._annotated_dataclass)
                sequences = signal_sequences if sequences is None else merge_dataclass_objects(
                    [sequences, signal_sequences])

        sequences = self._compute_p_gens_for_export(sequences, sim_item)

        sequences = make_receptor_sequence_objects(sequences,
                                                   metadata=make_signal_metadata(sim_item, self.state.signals),
                                                   immune_events=sim_item.immune_events,
                                                   custom_params=self._custom_fields,
                                                   chain=sim_item.generative_model.chain)

        return sequences

    def _compute_p_gens_for_export(self, sequences, sim_item: SimConfigItem):
        if self._export_p_gens:
            sequences = self._update_sequences_with_missing_p_gens(sequences, sim_item)
            if isinstance(self.state.simulation.simulation_strategy, ImplantingStrategy):
                sequences = self._update_sequences_with_missing_p_gens(sequences, sim_item,
                                                                       'original_sequence', 'original_p_gen')

        return sequences

    def _create_repertoires(self, item: SimConfigItem) -> list:
        path = PathBuilder.build(self.state.result_path / item.name)
        sequence_paths = self._gen_necessary_sequences(path, sim_item=item)

        repertoires = []
        used_seq_count = {**{'no_signal': 0}, **{signal.id: 0 for signal in item.signals}}
        repertoires_path = PathBuilder.build(path / "repertoires")

        for i in range(item.number_of_examples):
            seqs_no_signal_count = item.receptors_in_repertoire_count - sum(
                get_signal_sequence_count(1, proportion, item.receptors_in_repertoire_count)
                for _, proportion in item.signal_proportions.items())

            sequences, used_seq_count = get_signal_sequences(self._annotated_dataclass, used_seq_count, item,
                                                             sequence_paths)

            sequences, used_seq_count = get_no_signal_sequences(sequences=sequences, used_seq_count=used_seq_count,
                                                                seqs_no_signal_count=seqs_no_signal_count,
                                                                bnp_data_class=self._annotated_dataclass,
                                                                sequence_paths=sequence_paths,
                                                                sim_item=item)

            check_sequence_count(item, sequences)

            sequences = self._compute_p_gens_for_export(sequences, item)

            repertoire = make_repertoire_from_sequences(sequences, repertoires_path, item, self.state.signals,
                                                        self._custom_fields)
            repertoires.append(repertoire)

        return repertoires

    def _gen_necessary_sequences(self, base_path: Path, sim_item: SimConfigItem) -> Dict[str, Path]:

        if sim_item.seed is not None:
            np.random.seed(sim_item.seed)

        path = PathBuilder.build(base_path / sim_item.name)
        seqs_per_signal_count = get_sequence_per_signal_count(sim_item)
        seq_paths = make_sequence_paths(path, sim_item.signals)
        iteration = 1

        while sum(seqs_per_signal_count.values()) > 0 and iteration < self._max_iterations:
            sequences = self._make_background_sequences(path, iteration, sim_item, seqs_per_signal_count,
                                                        need_background_seqs=iteration == 1 and self.state.simulation.keep_p_gen_dist)

            if self.state.simulation.keep_p_gen_dist and sim_item.generative_model.can_compute_p_gens() and iteration == 1:
                self._make_p_gen_histogram(sequences, sim_item.name, path)
                print_log(
                    f"Computed a histogram from the first batch of background sequences for {sim_item.name}, available at: {str(path)}",
                    include_datetime=True)

            sequences = filter_sequences_by_length(sequences, sim_item, self.sequence_type)

            sequences = annotate_sequences(sequences, self.sequence_type == SequenceType.AMINO_ACID, self.state.signals,
                                           self._annotated_dataclass, sim_item.name)

            sequences = self.state.simulation.simulation_strategy.process_sequences(sequences, copy.deepcopy(
                seqs_per_signal_count), self._use_p_gens,  self.sequence_type, sim_item, self.state.signals,
                self.state.simulation.remove_seqs_with_signals,
                implanting_scaling_factor=self.state.simulation.implanting_scaling_factor)

            if sequences is not None and len(sequences) > 0:

                if self.state.simulation.keep_p_gen_dist and sim_item.generative_model.can_compute_p_gens():
                    sequences = self._filter_using_p_gens(sequences, sim_item)

                seqs_per_signal_count['no_signal'] = update_seqs_without_signal(seqs_per_signal_count['no_signal'],
                                                                                sequences, seq_paths['no_signal'])
                seqs_per_signal_count = update_seqs_with_signal(copy.deepcopy(seqs_per_signal_count), sequences,
                                                                self.state.signals, sim_item.signals,
                                                                seq_paths)

            print_log(
                f"Finished iteration {iteration} in {sim_item.name}: remaining sequence count per signal for {sim_item.name}: "
                f"{seqs_per_signal_count}" if sum(
                    seqs_per_signal_count.values()) > 0 else f"{sim_item.name} simulation finished", True)
            check_iteration_progress(iteration, self._max_iterations)
            iteration += 1

        if iteration == self._max_iterations and sum(seqs_per_signal_count.values()) != 0:
            raise SimError(
                f"{LigoSimInstruction.__name__}: maximum iterations were reached, but the simulation could not finish "
                f"with parameters: {vars(self.state.simulation)}.\n")

        return seq_paths

    def _make_background_sequences(self, path, iteration: int, sim_item: SimConfigItem, sequence_per_signal_count: dict,
                                   need_background_seqs: bool) -> BackgroundSequences:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{iteration}.tsv"

        v_genes = sorted(list(set(chain(signal.v_call for signal in sim_item.signals if signal.v_call is not None))))
        j_genes = sorted(list(set(chain(signal.j_call for signal in sim_item.signals if signal.j_call is not None))))

        if sequence_per_signal_count['no_signal'] > 0 or need_background_seqs or (
                len(v_genes) == 0 and len(j_genes) == 0) \
                or not sim_item.generative_model.can_generate_from_skewed_gene_models():
            sim_item.generative_model.generate_sequences(self._sequence_batch_size, seed=sim_item.seed,
                                                         path=sequence_path,
                                                         sequence_type=self.sequence_type,
                                                         compute_p_gen=self._use_p_gens)

            print_log(f"Generated {self._sequence_batch_size} background sequences, stored at {sequence_path}.", True)

        skew_model_for_signal = needs_seqs_with_signal(sequence_per_signal_count)

        if sim_item.generative_model.can_generate_from_skewed_gene_models() and skew_model_for_signal and (
                len(v_genes) > 0 or len(j_genes) > 0):
            sim_item.generative_model.generate_from_skewed_gene_models(v_genes=v_genes, j_genes=j_genes,
                                                                       seed=sim_item.seed, path=sequence_path,
                                                                       sequence_type=self.sequence_type,
                                                                       batch_size=self._sequence_batch_size,
                                                                       compute_p_gen=self._use_p_gens)

            print_log(
                f"Generated {self._sequence_batch_size} sequences from skewed model for given V/J genes at {sequence_path}.",
                True)

        data = get_bnp_data(sequence_path, BackgroundSequences)

        os.remove(sequence_path)
        print_log(f"Prepared sequences for processing and removed temporary file {sequence_path}.", True)

        return data

    def _make_p_gen_histogram(self, sequences: BackgroundSequences, sim_item_name: str, path: Path):

        log_p_gens = np.log10(sequences[sequences.from_default_model.astype(bool)].p_gen[
                                  np.nonzero(sequences[sequences.from_default_model.astype(bool)].p_gen)])

        hist, self.state.p_gen_bins[sim_item_name] = np.histogram(log_p_gens, density=False,
                                                                  bins=self.state.simulation.p_gen_bin_count)
        self.state.target_p_gen_histogram[sim_item_name] = hist / np.sum(sequences.from_default_model)

        zero_regions = self.state.target_p_gen_histogram[sim_item_name] == 0
        self.state.target_p_gen_histogram[sim_item_name][zero_regions] = ImplantingStrategy.MIN_RANGE_PROBABILITY
        self.state.target_p_gen_histogram[sim_item_name][np.logical_not(zero_regions)] -= \
            ImplantingStrategy.MIN_RANGE_PROBABILITY * (np.sum(zero_regions) + 1) / np.sum(np.logical_not(zero_regions))

        make_p_gen_histogram_plot(self.state.target_p_gen_histogram[sim_item_name],
                                  self.state.p_gen_bins[sim_item_name], path,
                                  str(LigoSimInstruction.MIN_RANGE_PROBABILITY))

    def _update_sequences_with_missing_p_gens(self, sequences: BackgroundSequences, sim_item: SimConfigItem,
                                              sequence_field: str = None, p_gen_field: str = "p_gen"):
        if np.any(getattr(sequences, p_gen_field) == -1):
            missing_p_gens = getattr(sequences, p_gen_field) == -1
            p_gens = getattr(sequences, p_gen_field)
            p_gens[missing_p_gens] = sim_item.generative_model.compute_p_gens(sequences[missing_p_gens], self.state.simulation.sequence_type, sequence_field)
            setattr(sequences, p_gen_field, p_gens)
        return sequences

    def _filter_using_p_gens(self, sequences: BackgroundSequences, sim_item: SimConfigItem) -> Tuple[
        BNPDataClass, dict]:
        sequences = self._update_sequences_with_missing_p_gens(sequences, sim_item)
        with np.errstate(divide='ignore'):
            p_gens = np.log10(sequences.p_gen)

        p_gen_bins = self.state.p_gen_bins[sim_item.name]
        hist = np.concatenate(
            [[LigoSimInstruction.MIN_RANGE_PROBABILITY], self.state.target_p_gen_histogram[sim_item.name],
             [LigoSimInstruction.MIN_RANGE_PROBABILITY]])

        seq_keep_prob = hist[np.digitize(p_gens, p_gen_bins)]
        keep_sequences = np.random.uniform(0, 1, len(sequences)) <= seq_keep_prob

        print_log(
            f"Removed {len(sequences) - sum(keep_sequences)} out of {len(sequences)} sequences from the batch when filtering by p_gens.",
            True)

        return sequences[keep_sequences]
