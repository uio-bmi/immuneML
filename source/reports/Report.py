import abc


class Report(metaclass=abc.ABCMeta):
    """
    This class defines what report classes should look like: they all have to inherit this class and implement
    generate_report method
    """

    @abc.abstractmethod
    def generate_report(self, params):
        pass
