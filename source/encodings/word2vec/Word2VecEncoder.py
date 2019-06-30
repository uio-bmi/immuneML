# quality: gold
import copy
import hashlib
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.preprocessing.FeatureScaler import FeatureScaler
from source.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from source.util.FilenameHandler import FilenameHandler
from source.util.KmerHelper import KmerHelper


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

    @staticmethod
    def encode(dataset: Dataset, params: EncoderParams) -> Dataset:
        encoded_dataset = CacheHandler.memo_by_params(Word2VecEncoder._prepare_caching_params(dataset, params),
                                                      lambda: Word2VecEncoder._encode_new_dataset(dataset, params))

        return encoded_dataset

    @staticmethod
    def _prepare_caching_params(dataset: Dataset, params: EncoderParams, vectors=None, description: str = ""):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("vectors", hashlib.sha256(str(vectors).encode("utf-8")).hexdigest()),
                ("description", description),
                ("encoding", Word2VecEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("encoding_params", tuple([(key, params["model"][key]) for key in params["model"].keys()])), )

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: EncoderParams) -> Dataset:
        if params["learn_model"] is True and not Word2VecEncoder._exists_model(params):
            model = Word2VecEncoder._create_model(dataset, params)
        else:
            model = Word2VecEncoder._load_model(params)

        vectors = model.wv
        del model

        encoded_dataset = Word2VecEncoder._encode_by_model(dataset, vectors, params)

        Word2VecEncoder.store(encoded_dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_repertoire(repertoire: Repertoire, vectors, params: EncoderParams):
        repertoire_vector = np.zeros(vectors.vector_size)
        for (index2, sequence) in enumerate(repertoire.sequences):
            kmers = KmerHelper.create_kmers_from_sequence(sequence=sequence, k=params["model"]["k"])
            sequence_vector = np.zeros(vectors.vector_size)
            for kmer in kmers:
                try:
                    word_vector = vectors.get_vector(kmer)
                    sequence_vector = np.add(sequence_vector, word_vector)
                except KeyError:
                    pass

            repertoire_vector = np.add(repertoire_vector, sequence_vector)
        return repertoire_vector

    @staticmethod
    def _encode_labels(dataset: Dataset, params: EncoderParams):

        label_config = params["label_configuration"]
        labels = {name: [] for name in label_config.get_labels_by_name()}

        for repertoire in dataset.get_data(params["batch_size"]):

            for label_name in label_config.get_labels_by_name():
                label = repertoire.metadata.custom_params[label_name]
                labels[label_name].append(label)

        return np.array([labels[name] for name in labels.keys()])

    @staticmethod
    def _encode_by_model(dataset, vectors, params: EncoderParams) -> Dataset:

        encoded_dataset = copy.deepcopy(dataset)

        repertoires = CacheHandler.memo_by_params(Word2VecEncoder._prepare_caching_params(dataset, params, vectors, Word2VecEncoder.DESCRIPTION_REPERTOIRES),
                                                  lambda: Word2VecEncoder._encode_repertoires(encoded_dataset, vectors, params))

        labels = CacheHandler.memo_by_params(Word2VecEncoder._prepare_caching_params(dataset, params, vectors, Word2VecEncoder.DESCRIPTION_LABELS),
                                             lambda: Word2VecEncoder._encode_labels(dataset, params))

        scaler_filename = params["result_path"] + FilenameHandler.get_filename("standard_scaling", "pkl")
        scaled_repertoires = FeatureScaler.standard_scale(scaler_filename, repertoires)

        encoded_dataset = Word2VecEncoder._build_encoded_dataset(encoded_dataset, scaled_repertoires, labels, params)
        return encoded_dataset

    @staticmethod
    def _build_encoded_dataset(encoded_dataset, scaled_repertoires, labels, params):

        label_names = params["label_configuration"].get_labels_by_name()
        feature_names = [str(i) for i in range(scaled_repertoires.shape[1])]
        feature_annotations = pd.DataFrame({"feature": feature_names})

        encoded_data = EncodedData(repertoires=scaled_repertoires,
                                   labels={label: labels[i] for i, label in enumerate(label_names)},
                                   repertoire_ids=[repertoire.identifier for repertoire in encoded_dataset.get_data()],
                                   feature_names=feature_names,
                                   feature_annotations=feature_annotations)

        encoded_dataset.add_encoded_data(encoded_data)
        return encoded_dataset

    @staticmethod
    def _encode_repertoires(encoded_dataset, vectors, params: EncoderParams):
        repertoires = np.zeros(shape=[encoded_dataset.get_repertoire_count(), vectors.vector_size])
        for (index, repertoire) in enumerate(encoded_dataset.get_data()):
            repertoires[index] = Word2VecEncoder._encode_repertoire(repertoire, vectors, params)
        return repertoires

    @staticmethod
    def _load_model(params: EncoderParams):
        model_path = Word2VecEncoder._create_model_path(params)
        model = Word2Vec.load(model_path)
        return model

    @staticmethod
    def _create_model(dataset: Dataset, params: EncoderParams):

        if params["model"]["model_creator"] == ModelType.SEQUENCE:
            model_creator = SequenceModelCreator()
        else:
            model_creator = KmerPairModelCreator()

        model = model_creator.create_model(dataset=dataset,
                                           params=params,
                                           model_path=Word2VecEncoder._create_model_path(params))

        return model

    @staticmethod
    def store(encoded_dataset: Dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])

    @staticmethod
    def _exists_model(params: EncoderParams) -> bool:
        return os.path.isfile(Word2VecEncoder._create_model_path(params))

    @staticmethod
    def _create_model_path(params: EncoderParams):
        return params["result_path"] + "W2V.model"

