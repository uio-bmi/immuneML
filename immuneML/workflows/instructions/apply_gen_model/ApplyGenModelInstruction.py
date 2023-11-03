from pathlib import Path

from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.workflows.instructions.GenModelInstruction import GenModelInstruction
from immuneML.workflows.instructions.GenModelInstruction import GenModelState


class ApplyGenModelState(GenModelState):
    pass


class ApplyGenModelInstruction(GenModelInstruction):
    """
    ApplyGenModel instruction implements applying generative AIRR models on receptor level.

    This instruction takes as input a trained model which will be used for generating data and the number of
    sequences to be generated. It can also produce reports of the applied model and reports of generated
    sequences.

    To train generative model with immuneML, see TrainGenModel instruction.

    Arguments:
        method (GenerativeModel): which model to apply
        gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model
        result_path (Path): path to the directory where the results will be stored
        name (str): name of the instruction
        reports (list): list of report ids (defined under definitions/reports) to apply after generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run on the fitted model)

    YAML specification:
        .. highlight:: yaml
        .. code-block:: yaml

        my_apply_gen_model_inst: # user-defined instruction name
            type: ApplyGenModel
            gen_examples_count: 100
            method: m1
            config_path: ./config.zip # path to the trained model in zip format (possibly output of TrainGenModel instruction)
            reports: [data_rep1, ml_rep2]

    """
    def __init__(self, method: GenerativeModel = None, reports: list = None, result_path: Path = None,
                 name: str = None, gen_examples_count: int = None):
        super().__init__(ApplyGenModelState(result_path, name, gen_examples_count), method, reports)

    def run(self, result_path: Path) -> GenModelState:
        self._set_path(result_path)
        self._gen_data()
        self._run_reports()

        return self.state
