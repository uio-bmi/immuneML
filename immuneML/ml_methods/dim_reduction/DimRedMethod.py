import abc


class DimRedMethod:

    def __init__(self, name: str):
        self.method = None
        self.name = name

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def transform(self, data):
        pass

    @classmethod
    def get_title(cls):
        return "Dimensionality "
