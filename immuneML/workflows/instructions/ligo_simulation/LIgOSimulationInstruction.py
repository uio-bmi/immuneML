import random
import shutil
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationStrategy import SimulationStrategy
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.ExporterHelper import ExporterHelper
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ligo_simulation.LIgOSimulationState import LIgOSimulationState


class LIgOSimulationInstruction(Instruction):
    """
    LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
    the user.

    Arguments:

        is_repertoire (bool): indicates if simulation should be on repertoire or receptor level

        paired (bool): if the simulated data should be paired or not

        use_generation_probabilities (bool): whether to base computations on generation probabilities of individual receptors

        simulation_strategy (str): how to construct receptors that contain a signal; possible options are `IMPLANTING` and `REJECTION_SAMPLING`

        simulation (str): a name of a simulation object containing a list of LIgOSimulationItem as specified under definitions key; defines how
        to combine signals with simulated data; specified under definitions

        store_signal_in_receptors (bool): for repertoire-level simulation, whether to store the information on what exact motif is implanted in each
        receptor

        export_formats: in which formats to export the dataset after simulation. Valid formats are class names of any non-abstract class
        inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`. Important note: Binary files in ImmuneML might not be compatible
        between different immuneML versions.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: LIgOSimulation # which instruction to execute
            is_repertoire: True
            paired: False
            use_generation_probabilities: False
            simulation_strategy: IMPLANTING
            simulation: sim1
            store_signal_in_receptors: True
            export_formats: [AIRR] # in which formats to export the dataset

    """

    def __init__(self, is_repertoire: bool, paired: bool, use_generation_probabilities: bool, simulation_strategy: SimulationStrategy,
                 simulation: Simulation, sequence_type: SequenceType, signals: List[Signal], name: str, store_signal_in_receptors: bool,
                 exporters: List[DataExporter] = None):

        self.state = LIgOSimulationState(is_repertoire=is_repertoire, paired=paired, use_generation_probabilities=use_generation_probabilities,
                                         simulation_strategy=simulation_strategy, simulation=simulation, sequence_type=sequence_type,
                                         signals=signals, name=name, store_signal_in_receptors=store_signal_in_receptors)
        self.exporters = exporters

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        examples = []

        for item in self.state.simulation.simulation_items:
            tmp_examples = self._create_examples(item)
            examples.extend(tmp_examples)

        labels = {signal.id: [True, False] for signal in self.state.signals}

        if self.state.is_repertoire:
            self.state.dataset = RepertoireDataset.build_from_objects(labels=labels, repertoires=examples, name='simulated_dataset',
                                                                      metadata_path=self.state.result_path / 'metadata.csv')
        elif self.state.paired:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        exporter_output = ExporterHelper.export_dataset(self.state.dataset, self.exporters, self.state.result_path)

        self.state.formats = exporter_output['formats']
        self.state.paths = exporter_output['paths']

        return self.state

    def _create_examples(self, item: LIgOSimulationItem) -> list:

        if self.state.is_repertoire:
            return self._create_repertoires(item)
        else:
            return self._create_receptors(item)

    def _create_receptors(self, item: LIgOSimulationItem):
        raise NotImplementedError

    def _create_repertoires(self, item: LIgOSimulationItem) -> list:

        repertoires = []
        repertoires_path = PathBuilder.build(self.state.result_path / "repertoires")

        for i in range(1, item.number_of_examples + 1):
            path = PathBuilder.build(self.state.result_path / f"tmp_background_repertoire_{i}/")

            receptors = self._make_receptors_with_signal(item, path=path)

            repertoire = Repertoire.build_from_sequence_objects(receptors, repertoires_path,
                                                                {**{signal.id: True for signal in item.signals},
                                                                 **{signal.id: False for signal in self.state.signals if signal not in item.signals}})

            repertoires.append(repertoire)

            shutil.rmtree(path)

        return repertoires

    def _make_receptors_with_signal(self, item: LIgOSimulationItem, path: Path) -> list:
        if self.state.simulation_strategy == SimulationStrategy.IMPLANTING:

            new_sequences = self._make_sequences_by_implanting(item=item, path=path)

        elif self.state.simulation_strategy == SimulationStrategy.REJECTION_SAMPLING:

            new_sequences = self._make_sequences_by_rejection(item=item, path=path)

        else:
            raise RuntimeError(f"{LIgOSimulationInstruction.__name__}: simulation strategy was not properly set, accepted are "
                               f"{SimulationStrategy.IMPLANTING.name} and {SimulationStrategy.REJECTION_SAMPLING.name}, but got "
                               f"{self.state.simulation_strategy} instead.")

        return new_sequences

    def _make_sequences_by_rejection(self, item, receptor_with_signal_count: int, path: Path) -> list:

        raise NotImplementedError()

        # batch_id = 1
        # receptors = []
        #
        # while len(receptors) < item.number_of_receptors_in_repertoire:
        #     background_sequences = item.generative_model.generate_sequences(item.number_of_receptors_in_repertoire, seed=1, path=path,
        #                                                                     sequence_type=self.state.sequence_type)
        #
        #     new_receptors = []
        #
        #     receptors.extend(new_receptors)
        #
        #     batch_id += 1
        #
        # return receptors

    def _make_sequences_by_implanting(self, item: LIgOSimulationItem, path: Path) -> list:

        background_sequences = item.generative_model.generate_sequences(item.number_of_receptors_in_repertoire, seed=1, path=path / "tmp.tsv",
                                                                        sequence_type=self.state.sequence_type)

        sequence_indices = {}
        available_indices = range(len(background_sequences))
        new_sequences = []

        for signal in item.signals:
            receptor_with_signal_count = signal.implanting_strategy.compute_implanting(item.repertoire_implanting_rate,
                                                                                       item.number_of_receptors_in_repertoire)
            sequence_indices[signal.id] = random.sample(available_indices, k=receptor_with_signal_count)
            available_indices = [ind for ind in available_indices if ind not in sequence_indices[signal.id]]
            for index in sequence_indices[signal.id]:
                implanted_sequence = signal.implant_in_sequence(sequence=background_sequences[index], is_noise=item.is_noise,
                                                                sequence_type=self.state.sequence_type)
                for other_signal in self.state.signals:
                    if other_signal.id != signal.id:
                        implanted_sequence.metadata.custom_params[other_signal.id] = False

                if not self.state.store_signal_in_receptors:
                    implanted_sequence.metadata.custom_params = {}
                    implanted_sequence.annotation = None

                new_sequences.append(implanted_sequence)

        for index in available_indices:
            background_sequence = background_sequences[index]

            if self.state.store_signal_in_receptors:
                background_sequence.metadata.custom_params = {**background_sequence.metadata.custom_params,
                                                              **{signal.id: False for signal in self.state.signals}}
            new_sequences.append(background_sequence)

        return new_sequences
