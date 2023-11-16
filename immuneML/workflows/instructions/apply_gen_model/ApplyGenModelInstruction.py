from pathlib import Path

from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.workflows.instructions.GenModelInstruction import GenModelInstruction
from immuneML.workflows.instructions.GenModelInstruction import GenModelState


class ApplyGenModelState(GenModelState):
    pass


class ApplyGenModelInstruction(GenModelInstruction):
    """
    ApplyGenModel instruction implements applying generative AIRR models on the sequence level.

    This instruction takes as input a trained model which will be used for generating data and the number of
    sequences to be generated. It can also produce reports of the applied model and reports of generated
    sequences.

    To train generative model with immuneML, see TrainGenModel instruction.

    .. note::

        This is an experimental feature in version 3.0.0a1.

    Specification arguments:

    - gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model

    - reports (list): list of report ids (defined under definitions/reports) to apply after generating
      gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
      on the fitted model)

    - config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

    YAML specification:

    .. highlight:: yaml
    .. code-block:: yaml

        my_apply_gen_model_inst: # user-defined instruction name
            type: ApplyGenModel
            gen_examples_count: 100
            ml_config_path: ./config.zip
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
