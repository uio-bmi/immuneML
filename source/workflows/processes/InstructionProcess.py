import abc


class InstructionProcess(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self):
        pass
