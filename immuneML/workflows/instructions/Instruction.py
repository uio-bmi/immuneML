import abc
from pathlib import Path


class Instruction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, result_path: Path):
        pass
