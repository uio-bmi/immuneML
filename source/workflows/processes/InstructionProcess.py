import abc


class InstructionProcess(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, result_path: str):
        pass
