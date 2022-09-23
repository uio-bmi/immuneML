import abc
from pathlib import Path

from sklearn.exceptions import NotFittedError

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment import Label


class GenerativeModel(metaclass=abc.ABCMeta):

    def __init__(self):
        self.name = None
        self.label = None

