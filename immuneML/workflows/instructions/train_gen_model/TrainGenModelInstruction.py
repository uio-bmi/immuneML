from dataclasses import dataclass
from pathlib import Path

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


@dataclass
class TrainGenModelState:
    result_path: Path
    name: str
    gen_examples_count: int
    sequence_examples: list = None
    model_path: Path = None


class TrainGenModelInstruction(Instruction):
    """
    TrainGenModel instruction implements training generative AIRR models on both repertoire and receptor level
    depending on the parameters and the chosen model. Models that can be trained for sequence generation are listed
    under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    or generated sequences.

    To use the generative model previously trained with immuneML, see ApplyGenModel instruction.

    Arguments:

        dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

        method: which model to fit (defined previously under definitions/ml_methods)

        number_of_processes (int): how many processes to use for fitting the model

        gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_train_gen_model_inst: # user-defined instruction name
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            model: model1 # defined previously under definitions/ml_methods
            gen_examples_count: 100
            number_of_processes: 4

    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, method: GenerativeModel = None, number_of_processes: int = 1,
                 gen_examples_count: int = 100, result_path: Path = None, name: str = None):
        self.dataset = dataset
        self.number_of_processes = number_of_processes
        self.method = method
        self.state = TrainGenModelState(result_path, name, gen_examples_count)

    def run(self, result_path: Path) -> TrainGenModelState:
        self._set_path(result_path)
        self._fit_model()
        self._save_model()
        self._gen_data()

        return self.state

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.method.fit(self.dataset)
        print_log(f"{self.state.name}: fitted the model", True)

    def _save_model(self):
        self.state.model_path = self.method.save_model(self.state.result_path / 'trained_model/')

    def _gen_data(self):
        dataset = self.method.generate_sequences(self.state.gen_examples_count, 1,
                                                 self.state.result_path / 'generated_sequences',
                                                 SequenceType.AMINO_ACID, False)

        print_log(f"{self.state.name}: generated {self.state.gen_examples_count} examples from the fitted model", True)

        AIRRExporter.export(dataset, self.state.result_path)

    def _set_path(self, result_path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
