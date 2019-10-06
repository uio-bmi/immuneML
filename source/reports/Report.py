import abc


class Report(metaclass=abc.ABCMeta):
    """
    This class defines what report classes should look like: they all have to inherit this class and implement
    generate_report method
    """

    @abc.abstractmethod
    def generate(self):
        pass

    def check_prerequisites(self):
        pass

    def set_context(self, context: dict):
        return self

    def generate_report(self):
        self.check_prerequisites()
        self.generate()
