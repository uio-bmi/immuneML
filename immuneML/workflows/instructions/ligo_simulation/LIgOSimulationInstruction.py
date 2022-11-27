import math
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationStrategy import SimulationStrategy
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.rejection_sampling.RejectionSampler import RejectionSampler
from immuneML.simulation.signal_implanting.LigoImplanter import LigoImplanter
from immuneML.util.ExporterHelper import ExporterHelper
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ligo_simulation.LIgOSimulationState import LIgOSimulationState


class LIgOSimulationInstruction(Instruction):
    """
    LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
    the user.

    Arguments:

        simulation (str): a name of a simulation object containing a list of LIgOSimulationItem as specified under definitions key; defines how
        to combine signals with simulated data; specified under definitions

        store_signal_in_receptors (bool): for repertoire-level simulation, whether to store the information on what exact motif is implanted in each receptor

        sequence_batch_size (bool): how many sequences to generate at once using the generative model before checking for signals and filtering

        max_iterations (int): how many iterations are allowed when creating sequences

        export_p_gens (bool): whether to compute generation probabilities (if supported by the generative model) for sequences and include them as part of output

        number_of_processes (int): determines how many simulation items can be simulated in parallel

        export_formats: in which formats to export the dataset after simulation. Valid formats are class names of any non-abstract class
        inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`. Important note: Binary files in ImmuneML might not be compatible
        between different immuneML versions.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: LIgOSimulation # which instruction to execute
            simulation: sim1
            store_signal_in_receptors: True
            sequence_batch_size: 1000
            max_iterations: 1000
            export_p_gens: False
            number_of_processes: 4
            export_formats: [AIRR] # in which formats to export the dataset

    """

    def __init__(self, simulation: Simulation, signals: List[Signal], name: str, store_signal_in_receptors: bool,
                 sequence_batch_size: int, max_iterations: int, number_of_processes: int, exporters: List[DataExporter] = None,
                 export_p_gens: bool = None):

        self.state = LIgOSimulationState(simulation=simulation, signals=signals, name=name, store_signal_in_receptors=store_signal_in_receptors,
                                         number_of_processes=number_of_processes, sequence_batch_size=sequence_batch_size,
                                         max_iterations=max_iterations)
        self.exporters = exporters
        self.export_p_gens = export_p_gens

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        self._simulate_dataset()

        exporter_output = ExporterHelper.export_dataset(self.state.dataset, self.exporters, self.state.result_path)

        self.state.formats = exporter_output['formats']
        self.state.paths = exporter_output['paths']

        return self.state

    def _simulate_dataset(self):
        chunk_size = math.ceil(len(self.state.simulation.sim_items) / self.state.number_of_processes)

        with Pool(processes=self.state.number_of_processes) as pool:
            result = pool.map(self._create_examples, [vars(item) for item in self.state.simulation.sim_items], chunksize=chunk_size)
            examples = list(chain.from_iterable(result))

        labels = {signal.id: [True, False] for signal in self.state.signals}

        if self.state.simulation.is_repertoire:
            self.state.dataset = RepertoireDataset.build_from_objects(labels=labels, repertoires=examples, name='simulated_dataset',
                                                                      metadata_path=self.state.result_path / 'metadata.csv')
        elif self.state.simulation.paired:
            raise NotImplementedError()
        else:
            self.state.dataset = SequenceDataset.build_from_objects(examples, path=self.state.result_path, name='simulated_dataset',
                                                                    file_size=SequenceDataset.DEFAULT_FILE_SIZE, labels=labels)

    def _create_examples(self, item_in: dict) -> list:

        item = LIgOSimulationItem(**item_in)

        if self.state.simulation.is_repertoire:
            res = self._create_repertoires(item)
        else:
            res = self._create_receptors(item)

        return res

    def _create_receptors(self, item: LIgOSimulationItem):
        if self.state.simulation.paired:
            raise NotImplementedError
        else:
            if self.state.simulation.simulation_strategy == SimulationStrategy.REJECTION_SAMPLING:
                sampler = RejectionSampler(sim_item=item, sequence_type=self.state.simulation.sequence_type, all_signals=self.state.signals,
                                           seed=item.seed, sequence_batch_size=self.state.sequence_batch_size,
                                           max_iterations=self.state.max_iterations, export_pgens=self.export_p_gens)
                sequences = sampler.make_sequences(self.state.result_path)
                return sequences
            else:
                raise NotImplementedError

    def _create_repertoires(self, item: LIgOSimulationItem) -> list:

        if self.state.simulation.simulation_strategy == SimulationStrategy.IMPLANTING:

            repertoires = LigoImplanter(item, self.state.simulation.sequence_type, self.state.signals, self.state.sequence_batch_size, item.seed,
                                        self.export_p_gens).make_repertoires(self.state.result_path)

        elif self.state.simulation.simulation_strategy == SimulationStrategy.REJECTION_SAMPLING:

            sampler = RejectionSampler(sim_item=item, sequence_type=self.state.simulation.sequence_type, all_signals=self.state.signals,
                                       seed=item.seed, sequence_batch_size=self.state.sequence_batch_size, max_iterations=self.state.max_iterations,
                                       export_pgens=self.export_p_gens)
            repertoires = sampler.make_repertoires(self.state.result_path)

        else:
            raise RuntimeError(f"{LIgOSimulationInstruction.__name__}: simulation strategy was not properly set, accepted are "
                               f"{SimulationStrategy.IMPLANTING.name} and {SimulationStrategy.REJECTION_SAMPLING.name}, but got "
                               f"{self.state.simulation.simulation_strategy} instead.")

        return repertoires

    def _add_to_summary(self, summary_path, receptor_with_signal_count, signal, item, repertoire_id):

        df = pd.DataFrame({"receptors_with_signal": [receptor_with_signal_count],
                           "receptors_total": [item.receptors_in_repertoire_count],
                           "signal_id": [signal.id], "simulation_item": [item.name],
                           "repertoire_id": [repertoire_id]})

        if summary_path.is_file():
            df.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_path, mode='w', index=False)
