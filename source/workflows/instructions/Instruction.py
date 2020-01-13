import abc


class Instruction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, result_path: str):
        pass
