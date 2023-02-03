
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.PathBuilder import PathBuilder


class TextAsIntEncoder(DatasetEncoder):

    def __init__(self, name: str = None):
        self.name = name
        self.max_sequence_length = 0

    def _prepare_parameters(name: str = None):
        return {"name": name}

    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = TextAsIntEncoder._prepare_parameters(**params)
            return TextAsIntEncoder(**prepared_params)
        else:
            raise ValueError("TextAsIntEncoder is not defined for dataset types which are not RepertoireDataset.")

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file if hasattr(dataset, "metadata_file") else None),
                ("dataset_type", dataset.__class__.__name__),
                ("encoding", TextAsIntEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", step),
                ("encoding_params", tuple(vars(self).items())))

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = RepertoireDataset(repertoires=dataset.repertoires,
                                            encoded_data=encoded_data,
                                            labels=dataset.labels,
                                            metadata_file=dataset.metadata_file)

        return encoded_dataset



    def _encode_data(self, dataset, params: EncoderParams):

        #Creates one long string containing every single sequence
        instances = ' '.join(
            [(sequence.get_sequence()) for repertoire in dataset.get_data() for sequence in repertoire.sequences])


        length_of_sequence = 21 #hardcoded length of the sequence, consider making different

        info = {"length_of_sequence": length_of_sequence}

        encoded_data = EncodedData(examples=instances,
                                   #feature_names=feature_names,
                                   encoding=TextAsIntEncoder.__name__,
                                   info=info)

        return encoded_data




