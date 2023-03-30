
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class CharToIntEncoder(DatasetEncoder):

    def __init__(self, sequence_type: SequenceType, name: str = None):
        self.name = name
        self.sequence_type = sequence_type

    def _prepare_parameters(sequence_type: str, name: str = None):
        return {"name": name, "sequence_type": SequenceType(sequence_type)}

    def build_object(dataset, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = CharToIntEncoder._prepare_parameters(**params)
            return CharToIntEncoder(**prepared_params)
        else:
            raise ValueError("ChartToInt is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams) -> SequenceDataset:

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file if hasattr(dataset, "metadata_file") else None),
                ("dataset_type", dataset.__class__.__name__),
                ("encoding", CharToIntEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", step),
                ("encoding_params", tuple(vars(self).items())))

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = SequenceDataset(encoded_data=encoded_data, labels=dataset.labels)

        return encoded_dataset

    def _encode_data(self, dataset, params: EncoderParams):

        sequences = dataset.get_data()
        all_sequences = ""
        for sequence in sequences:
            all_sequences += sequence.get_sequence() + " "

        alphabet = EnvironmentSettings.get_sequence_alphabet(self.sequence_type)
        alphabet.append(" ")
        char2idx = {u: i for i, u in enumerate(alphabet)}

        examples = [char2idx[c] for c in all_sequences]

        encoded_data = EncodedData(examples=examples,
                                   encoding=CharToIntEncoder.__name__)

        return encoded_data




