from dataclasses import dataclass
from pathlib import Path

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


@dataclass
class TrainGenModelState:
    result_path: Path
    name: str
    gen_sequence_count: int
    sequence_examples: list = None


class TrainGenModelInstruction(Instruction):
    """
    Some docs for TrainGenModel
    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, model: GenerativeModel = None, number_of_processes: int = 1,
                 gen_sequence_count: int = 100, result_path: Path = None, name: str = None):
        self.dataset = dataset
        self.number_of_processes = number_of_processes
        self.model = model
        self.state = TrainGenModelState(result_path, name, gen_sequence_count)

    def run(self, result_path: Path) -> TrainGenModelState:
        self._set_path(result_path)
        self._fit_model()
        self._save_model()
        self._gen_data()

        return self.state

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.model.fit(self.dataset)
        print_log(f"{self.state.name}: fitted the model", True)

    def _save_model(self):
        print(self.model)

    def _gen_data(self):
        dataset = self.model.generate_sequences(self.state.gen_sequence_count, 1,
                                                self.state.result_path / 'generated_sequences',
                                                SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated sample sequences from the fitted model", True)

        AIRRExporter.export(dataset, self.state.result_path)

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
