from dataclasses import dataclass
from pathlib import Path
from typing import List

from bionumpy.io.motifs import read_motif
from bionumpy.sequence.position_weight_matrix import PWM as bnp_PWM

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif


@dataclass
class LigoPWM(Motif):
    """
    Class describing positional weight matrix and using bionumpy's LigoPWM internally.
    For more details on bionumpy's implementation of PWM, see the documentation at
    https://bionumpy.github.io/bionumpy/tutorials/position_weight_matrix.html.

    Arguments:

        file_path: path to the file where the LigoPWM is stored

        threshold (float): when matching LigoPWM to a sequence, this is the threshold to consider the sequence as containing the motif

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_custom_pwm: # this will be the identifier of the motif
                file_path: my_pwm_1.csv
                threshold: 2

    """
    file_path: Path
    pwm_matrix: bnp_PWM
    threshold: float

    @classmethod
    def build(cls, identifier: str, file_path, threshold: float):
        assert Path(file_path).is_file(), file_path
        pwm_matrix = read_motif(file_path)
        return LigoPWM(identifier, file_path, pwm_matrix, threshold)

    def get_all_possible_instances(self, sequence_type: SequenceType):
        assert sorted(self.pwm_matrix.alphabet) == sorted(EnvironmentSettings.get_sequence_alphabet(sequence_type))
        return self

    def get_max_length(self) -> int:
        return self.pwm_matrix.window_size

    def get_alphabet(self) -> List[str]:
        return list(self.pwm_matrix.alphabet)

    def instantiate_motif(self, sequence_type: SequenceType = SequenceType.AMINO_ACID):
        pass
