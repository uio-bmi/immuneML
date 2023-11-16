from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.ReportResult import ReportResult
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.util import annotate_sequences, make_annotated_dataclass, get_bnp_data
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ligo_sim_feasibility.feasibility_reports import report_signal_frequencies, \
    report_signal_cooccurrences, \
    report_p_gen_histogram, report_seq_len_dist, report_signal_cond_probs, report_signal_joint_probs


@dataclass
class FeasibilitySumReports:
    signal_frequencies: ReportResult = None
    signal_cooccurrences: ReportResult = None
    signal_cond_probs: ReportResult = None
    signal_joint_probs: ReportResult = None
    p_gen_histogram: ReportResult = None
    seq_len_dist: ReportResult = None
    warnings: list = field(default_factory=list)


@dataclass
class FeasibilitySummaryState:
    simulation: SimConfig
    sequence_count: int
    signals: List[Signal]
    name: str = None
    result_path: Path = None
    reports: Dict[str, FeasibilitySumReports] = field(default_factory=dict)


class FeasibilitySummaryInstruction(Instruction):
    """
    FeasibilitySummary instruction creates a small synthetic dataset and reports summary metrics to show if the simulation with the given
    parameters is feasible. The input parameters to this analysis are the name of the simulation
    (the same that can be used with LigoSim instruction later if feasibility analysis looks acceptable), and the number of sequences to
    simulate for estimating the feasibility.

    The feasibility analysis is performed for each generative model separately as these could differ in the analyses that will be reported.

    Specification arguments:

    - simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

    - sequence_count (int): how many sequences to generate to estimate feasibility (default value: 100 000)

    - number_of_processes (int): for the parts of the analysis that are possible to parallelize, how many processes to use

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_feasibility_summary: # user-defined name of the instruction
            type: FeasibilitySummary # which instruction to execute
            simulation: sim1
            sequence_count: 10000

    """

    MIN_SIG_FREQ = 1e-5
    MAX_SIG_FREQ = 0.1

    def __init__(self, simulation, sequence_count: int, number_of_processes: int, signals: List[Signal],
                 name: str = None):
        self.state = FeasibilitySummaryState(simulation=simulation, sequence_count=sequence_count, signals=signals,
                                             name=name)
        self._number_of_processes = number_of_processes

        self._annotation_fields = sorted(
            [(signal.id, int) for signal in self.state.signals] + [('signals_aggregated', str)] +
            [(f"{signal.id}_positions", str) for signal in self.state.signals],
            key=lambda x: x[0])

        self._annotated_dc = make_annotated_dataclass(self._annotation_fields, self.state.signals)

    def run(self, result_path: Path):

        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        unique_models = self._get_unique_gen_models()

        for model_name, model in unique_models.items():
            self._make_summary(model, PathBuilder.build(self.state.result_path / model_name), model_name)

        return self.state

    def _make_summary(self, model: GenerativeModel, summary_path: Path, model_name: str):
        sequences = self._make_sequences(model, summary_path / "receptors.tsv", model_name)

        self.state.reports[model_name] = FeasibilitySumReports()
        self._make_signal_frequencies(sequences, summary_path / 'signal_frequencies', model_name)
        self._report_signal_co_occurrence(sequences, summary_path / 'signal_cooccurrence', model_name)
        self._make_pgen_dist(sequences, summary_path / 'p_gen_distribution', model, model_name)
        self._make_sequence_length_dist(sequences, summary_path / 'seq_length_distribution', model_name)

    def _make_sequence_length_dist(self, sequences: BackgroundSequences, path: Path, model_name: str):
        PathBuilder.build(path)
        self.state.reports[model_name].seq_len_dist = report_seq_len_dist(sequences,
                                                                          self.state.simulation.sequence_type, path)

        print_log(f"Estimated sequence length distribution for model {model_name}.", include_datetime=True)

    def _make_signal_frequencies(self, sequences: BackgroundSequences, path: Path, model_name: str):
        if len(self.state.signals) > 0:
            frequencies = pd.DataFrame({'signal': [signal.id for signal in self.state.signals],
                                        'frequency': [getattr(sequences, signal.id).sum() / len(sequences) for signal in
                                                      self.state.signals]})

            self.state.reports[model_name].signal_frequencies = report_signal_frequencies(frequencies,
                                                                                          PathBuilder.build(path))

            self._add_low_freq_warning(frequencies, model_name)
            self._add_high_freq_warning(frequencies, model_name)

            print_log(f"Computed signal frequencies for {len(self.state.signals)} signals for model {model_name}.",
                      include_datetime=True)

    def _add_high_freq_warning(self, frequencies: pd.DataFrame, model_name: str):
        high_frequencies = frequencies[frequencies['frequency'] >= FeasibilitySummaryInstruction.MAX_SIG_FREQ]
        for _, row in high_frequencies.iterrows():
            seqs_without_signal = self.state.simulation.get_total_seq_count(
                model_name) - self.state.simulation.get_total_seq_count_for_signal(row['signal'], model_name)
            if row['frequency'] == 1:
                self.state.reports[model_name].warnings.append(
                    f"Signal {row['signal']} occurs in all simulated sequences, but the simulation requires "
                    f"{seqs_without_signal} sequence(s) without the signal. Try feasibility analysis on larger batch "
                    f"size (currently {self.state.sequence_count}) or try updating the signal.")
            else:
                seqs_to_simulate = round(seqs_without_signal / (1 - row['frequency']))
                self.state.reports[model_name].warnings.append(
                    f"Signal {row['signal']} has very high frequency. It is found in "
                    f"{round(row['frequency'] * self.state.sequence_count)} out of {self.state.sequence_count} sequences. "
                    f"To simulate {seqs_without_signal} that do not contain the signal, it will require roughly "
                    f"{seqs_to_simulate} to be generated for rejection sampling. With the current batch size "
                    f"({self.state.sequence_count}), it will require roughly "
                    f"{round(seqs_to_simulate / self.state.sequence_count)} batch(es).")

    def _add_low_freq_warning(self, frequencies: pd.DataFrame, model_name: str):
        low_frequencies = frequencies[frequencies['frequency'] <= FeasibilitySummaryInstruction.MIN_SIG_FREQ]
        for _, row in low_frequencies.iterrows():
            seq_count_for_signal = self.state.simulation.get_total_seq_count_for_signal(row['signal'], model_name)
            if row['frequency'] == 0:
                self.state.reports[model_name].warnings.append(
                    f"Signal {row['signal']} does not occur in any of the simulated sequences, but the simulation "
                    f"requires {seq_count_for_signal} sequence(s) with the signal. Try feasibility analysis on larger "
                    f"batch size (currently {self.state.sequence_count}) or try updating the signal.")
            else:
                seqs_to_simulate = round(seq_count_for_signal / row['frequency'])
                self.state.reports[model_name].warnings.append(
                    f"Signal {row['signal']} has very low frequency. It is found in "
                    f"{round(row['frequency'] * self.state.sequence_count)} out of {self.state.sequence_count} sequences. "
                    f"To simulate {seq_count_for_signal} such sequence(s), it will require roughly {seqs_to_simulate} "
                    f"sequence(s) to be generated for rejection sampling. With the current batch size "
                    f"({self.state.sequence_count}), it will require roughly "
                    f"{round(seqs_to_simulate / self.state.sequence_count)} batch(es).")

    def _report_signal_co_occurrence(self, sequences: BackgroundSequences, path: Path, model_name: str):
        if len(self.state.signals) > 0:
            PathBuilder.build(path)
            unique_values, counts = np.unique(sequences.get_signal_matrix().sum(axis=1).reshape(-1, 1),
                                              return_counts=True)
            self.state.reports[model_name].signal_cooccurrences = report_signal_cooccurrences(unique_values, counts,
                                                                                              path)
            self.state.reports[model_name].signal_joint_probs = report_signal_joint_probs(sequences.get_signal_matrix(),
                                                                                          sequences.get_signal_names(),
                                                                                          path)
            self.state.reports[model_name].signal_cond_probs = report_signal_cond_probs(sequences.get_signal_matrix(),
                                                                                        sequences.get_signal_names(),
                                                                                        path)

            print_log(f"Examined signal co-occurrences for {len(self.state.signals)} signals for model {model_name}.",
                      include_datetime=True)

    def _make_pgen_dist(self, sequences: BackgroundSequences, path: Path, model: GenerativeModel, model_name: str):
        if self.state.simulation.keep_p_gen_dist and model.can_compute_p_gens() and self.state.simulation.p_gen_bin_count > 0:
            PathBuilder.build(path)
            self.state.reports[model_name].p_gen_histogram = report_p_gen_histogram(sequences,
                                                                                    self.state.simulation.p_gen_bin_count,
                                                                                    path)

            self.state.reports[model_name].warnings.append(
                "This simulation relies on using generation probabilities, which can significantly slow down the simulation, especially for large dataset sizes.")

            print_log(f"Estimated generation probability distribution for model {model_name}.", include_datetime=True)

    def _make_sequences(self, model, path: Path, model_name: str) -> BackgroundSequences:
        seq_path = model.generate_sequences(self.state.sequence_count, seed=0, path=path,
                                            sequence_type=self.state.simulation.sequence_type,
                                            compute_p_gen=model.can_compute_p_gens() and self.state.simulation.keep_p_gen_dist and self.state.simulation.p_gen_bin_count > 0)

        default_seqs = get_bnp_data(seq_path, BackgroundSequences)
        default_seqs = annotate_sequences(default_seqs, self.state.simulation.sequence_type == SequenceType.AMINO_ACID,
                                          self.state.signals,
                                          self._annotated_dc, model_name)

        print_log(f"Generated and annotated {self.state.sequence_count} sequences for model {model_name}",
                  include_datetime=True)

        return default_seqs

    def _get_unique_gen_models(self) -> Dict[str, GenerativeModel]:
        unique_models = {}

        for sim_item in self.state.simulation.sim_items:
            if not any(sim_item.generative_model.is_same(model) for model in unique_models.values()):

                unique_models[sim_item.name] = sim_item.generative_model

            else:
                new_model_name, old_model_name = None, None
                i = 0
                while old_model_name is None and i < len(unique_models.keys()):
                    model_name = list(unique_models.keys())[i]
                    model = unique_models[model_name]
                    if sim_item.generative_model.is_same(model):
                        new_model_name = "__".join([model_name, sim_item.name])
                        old_model_name = model_name
                    i += 1

                unique_models[new_model_name] = unique_models[old_model_name]
                del unique_models[old_model_name]

        return unique_models
