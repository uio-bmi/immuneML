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
        """
        Checks prerequisites, and sends warnings if the prerequisites are incorrect.
        Returns boolean value True if the prerequisites are o.k., and False otherwise.
        """
        return True

    def set_context(self, context: dict):
        return self

    def generate_report(self):
        if self.check_prerequisites():
            self.generate()
