import abc


class DatasetItem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_attribute(self, name: str):
        pass
