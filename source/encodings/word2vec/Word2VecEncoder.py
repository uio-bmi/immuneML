# quality: gold
import abc
import copy
import hashlib
import os

import pandas as pd
from gensim.models import Word2Vec

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.preprocessing.FeatureScaler import FeatureScaler
from source.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from source.util.FilenameHandler import FilenameHandler
from source.util.ReflectionHandler import ReflectionHandler


class Word2VecEncoder(DatasetEncoder):
    """
    Encodes the dataset using Word2Vec model.
    It relies on gensim's implementation of Word2Vec and KmerHelper for k-mer extraction.

    Configuration parameters are an instance of EncoderParams class:
    {
        "model": {
            "k": 3,
            "model_creator": ModelType.SEQUENCE,
            "size": 16
        },
        "batch_size": 1,
        "learn_model": True, # true for training set and false for test set
        "result_path": "../",
        "label_configuration": LabelConfiguration(), # labels should be set before encodings is invoked
    }

    NB: In order to use the workers properly and be able to parallelize the training process,
    it is necessary that Cython is installed on the machine.
    """

    DESCRIPTION_REPERTOIRES = "repertoires"
    DESCRIPTION_LABELS = "labels"

    dataset_mapping = {
        "RepertoireDataset": "W2VRepertoireEncoder"
    }

    @staticmethod
    def create_encoder(dataset=None):
        return ReflectionHandler.get_class_by_name(Word2VecEncoder.dataset_mapping[dataset.__class__.__name__], "word2vec/")()

    def encode(self, dataset, params: EncoderParams):
        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params, vectors=None, description: str = ""):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("vectors", hashlib.sha256(str(vectors).encode("utf-8")).hexdigest()),
                ("description", description),
                ("encoding", Word2VecEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("encoding_params", tuple([(key, params["model"][key]) for key in params["model"].keys()])), )

    def _encode_new_dataset(self, dataset, params):
        if params["learn_model"] is True and not self._exists_model(params):
            model = self._create_model(dataset=dataset, params=params)
        else:
            model = self._load_model(params)

        vectors = model.wv
        del model

        encoded_dataset = self._encode_by_model(dataset, params, vectors)

        self.store(encoded_dataset, params)

        return encoded_dataset

    @abc.abstractmethod
    def _encode_labels(self, dataset, params: EncoderParams):
        pass

    def _encode_by_model(self, dataset, params: EncoderParams, vectors):
        examples = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params, vectors,
                                                                            Word2VecEncoder.DESCRIPTION_REPERTOIRES),
                                               lambda: self._encode_examples(dataset, vectors, params))

        labels = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params, vectors, Word2VecEncoder.DESCRIPTION_LABELS),
                                             lambda: self._encode_labels(dataset, params))

        scaler_filename = params["result_path"] + FilenameHandler.get_filename("standard_scaling", "pkl")
        scaled_examples = FeatureScaler.standard_scale(scaler_filename, examples)

        encoded_dataset = self._build_encoded_dataset(dataset, scaled_examples, labels, params)
        return encoded_dataset

    def _build_encoded_dataset(self, dataset, scaled_examples, labels, params: EncoderParams):

        encoded_dataset = copy.deepcopy(dataset)

        label_names = params["label_configuration"].get_labels_by_name()
        feature_names = [str(i) for i in range(scaled_examples.shape[1])]
        feature_annotations = pd.DataFrame({"feature": feature_names})

        encoded_data = EncodedData(examples=scaled_examples,
                                   labels={label: labels[i] for i, label in enumerate(label_names)},
                                   example_ids=[example.identifier for example in encoded_dataset.get_data()],
                                   feature_names=feature_names,
                                   feature_annotations=feature_annotations,
                                   encoding=Word2VecEncoder.__name__)

        encoded_dataset.add_encoded_data(encoded_data)
        return encoded_dataset

    @abc.abstractmethod
    def _encode_examples(self, encoded_dataset, vectors, params):
        pass

    def _load_model(self, params):
        model_path = self._create_model_path(params)
        model = Word2Vec.load(model_path)
        return model

    def _create_model(self, dataset, params):

        if params["model"]["model_creator"] == ModelType.SEQUENCE:
            model_creator = SequenceModelCreator()
        else:
            model_creator = KmerPairModelCreator()

        model = model_creator.create_model(dataset=dataset,
                                           params=params,
                                           model_path=self._create_model_path(params))

        return model

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])

    def _exists_model(self, params: EncoderParams) -> bool:
        return os.path.isfile(self._create_model_path(params))

    def _create_model_path(self, params: EncoderParams):
        return params["result_path"] + "W2V.model"
