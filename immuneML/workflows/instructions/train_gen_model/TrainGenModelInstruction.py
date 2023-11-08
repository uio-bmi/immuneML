from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import print_log
from immuneML.workflows.instructions.GenModelInstruction import GenModelState, GenModelInstruction


class TrainGenModelState(GenModelState):
    pass


class TrainGenModelInstruction(GenModelInstruction):
    """
    TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
    for sequence generation are listed under Generative Models section.

    This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
    sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
    and reports of original and generated sequences.

    To use the generative model previously trained with immuneML, see ApplyGenModel instruction.

    .. note::

        This is an experimental feature in version 3.0.0a1.

    Specification arguments:

    - dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

    - method: which model to fit (defined previously under definitions/ml_methods)

    - number_of_processes (int): how many processes to use for fitting the model

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

    - reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
      and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
      reports (to be run on the fitted model)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_train_gen_model_inst: # user-defined instruction name
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            model: model1 # defined previously under definitions/ml_methods
            gen_examples_count: 100
            number_of_processes: 4
            reports: [data_rep1, ml_rep2]

    """

    MAX_ELEMENT_COUNT_TO_SHOW = 10

    def __init__(self, dataset: Dataset = None, method: GenerativeModel = None, number_of_processes: int = 1,
                 gen_examples_count: int = 100, result_path: Path = None, name: str = None, reports: list = None):
        super().__init__(TrainGenModelState(result_path, name, gen_examples_count), method, reports)
        self.dataset = dataset
        self.number_of_processes = number_of_processes

    def run(self, result_path: Path) -> GenModelState:
        self._set_path(result_path)
        self._fit_model()
        self._save_model()
        self._gen_data()
        self._evaluate_model()
        self._run_reports()

        return self.state

    def _fit_model(self):
        print_log(f"{self.state.name}: starting to fit the model", True)
        self.method.fit(self.dataset, self.state.result_path)
        print_log(f"{self.state.name}: fitted the model", True)

    def _save_model(self):
        self.state.model_path = self.method.save_model(self.state.result_path / 'trained_model/')

    def _evaluate_model(self):
        print("Evaluation is not implemented yet!")
