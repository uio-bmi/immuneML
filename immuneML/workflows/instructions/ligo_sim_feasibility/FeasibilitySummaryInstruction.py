from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportResult import ReportResult
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.util import annotate_sequences, make_annotated_dataclass, get_bnp_data
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ligo_sim_feasibility.feasibility_reports import report_signal_frequencies, report_signal_cooccurrences, \
    make_p_gen_histogram, report_p_gen_histogram


@dataclass
class FeasibilitySumReports:
    signal_frequencies: ReportResult = None
    signal_cooccurrences: ReportResult = None
    p_gen_histogram: ReportResult = None


@dataclass
class FeasibilitySummaryState:
    simulation: SimConfig
    sequence_count: int
    signals: List[Signal]
    name: str = None
    reports: FeasibilitySumReports = field(default_factory=FeasibilitySumReports)


class FeasibilitySummaryInstruction(Instruction):
    """
        FeasibilitySummaryInstruction instruction creates a small synthetic dataset and reports summary metrics to show if the simulation with the given
        parameters is feasible. The input parameters to this analysis are the name of the simulation
        (the same that can be used with LigoSim instruction later if feasibility analysis looks acceptable), and the number of sequences to
        simulate for estimating the feasibility.

        The feasibility analysis is performed for each generative model separately as these could differ in the analyses that will be reported.

        Arguments:

            simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how
            to combine signals with simulated data; specified under definitions

            sequence_count (int): how many sequences to generate to estimate feasibility (default value: 100 000)

            number_of_processes (int): for the parts of analysis that are possible to parallelize, how many processes to use

        YAML specification:

        .. indent with spaces
        .. code-block:: yaml

            my_feasibility_summary: # user-defined name of the instruction
                type: FeasibilitySummaryInstruction # which instruction to execute
                simulation: sim1
                sequence_count: 10000

        """

    def __init__(self, simulation, sequence_count: int, number_of_processes: int, signals: List[Signal], name: str = None):
        self.state = FeasibilitySummaryState(simulation=simulation, sequence_count=sequence_count, signals=signals, name=name)
        self._number_of_processes = number_of_processes

        self._annotation_fields = sorted([(signal.id, int) for signal in self.state.signals] +
                                         [(f"{signal.id}_positions", str) for signal in self.state.signals],
                                         key=lambda x: x[0])

        self._annotated_dc = make_annotated_dataclass(self._annotation_fields, self.state.signals)

    def run(self, result_path: Path):

        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        unique_models = self._get_unique_gen_models()

        for model_name, model in unique_models.items():
            self._make_summary(model, PathBuilder.build(self.state.result_path / model_name), model_name)

    def _make_summary(self, model: GenerativeModel, summary_path: Path, model_name: str):
        sequences = self._make_sequences(model, summary_path / "receptors.tsv")

        self._make_signal_frequencies(sequences, summary_path / 'signal_frequencies')
        self._report_signal_co_occurrence(sequences, summary_path / 'signal_cooccurrence')
        self._make_pgen_dist(sequences, summary_path / 'p_gen_distribution', model)

    def _make_signal_frequencies(self, sequences: BackgroundSequences, path: Path):
        if len(self.state.signals) > 0:
            frequencies = pd.DataFrame({'signal': [signal.id for signal in self.state.signals],
                                        'frequency': [round(getattr(sequences, signal.id).sum() / len(sequences), 2) for signal in
                                                      self.state.signals]})

            self.state.reports.signal_frequencies = report_signal_frequencies(frequencies, PathBuilder.build(path))

    def _report_signal_co_occurrence(self, sequences: BackgroundSequences, path: Path):
        if len(self.state.signals) > 0:
            PathBuilder.build(path)
            unique_values, counts = np.unique(sequences.get_signal_matrix().sum(axis=1).reshape(-1, 1), return_counts=True)
            self.state.reports.signal_cooccurrences = report_signal_cooccurrences(unique_values, counts, path)

    def _make_pgen_dist(self, sequences: BackgroundSequences, path: Path, model: GenerativeModel):
        if self.state.simulation.keep_p_gen_dist and model.can_compute_p_gens() and self.state.simulation.p_gen_bin_count > 0:
            PathBuilder.build(path)
            histogram, bins = make_p_gen_histogram(sequences, self.state.simulation.p_gen_bin_count)
            self.state.reports.p_gen_histogram = report_p_gen_histogram(histogram, bins, path)

    def _make_sequences(self, model, path: Path) -> BackgroundSequences:
        seq_path = model.generate_sequences(self.state.sequence_count, seed=0, path=path, sequence_type=self.state.simulation.sequence_type,
                                            compute_p_gen=model.can_compute_p_gens())

        default_seqs = get_bnp_data(seq_path, BackgroundSequences)
        default_seqs = annotate_sequences(default_seqs, self.state.simulation.sequence_type == SequenceType.AMINO_ACID, self.state.signals,
                                          self._annotated_dc)

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
                        new_model_name = "_".join([model_name, sim_item.name])
                        old_model_name = model_name
                    i += 1

                unique_models[new_model_name] = unique_models[old_model_name]
                del unique_models[old_model_name]

        return unique_models
