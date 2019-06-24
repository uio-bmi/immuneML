# quality: gold
import copy
import os
import pickle

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from source.util.FilenameHandler import FilenameHandler
from source.util.KmerHelper import KmerHelper
from source.util.PathBuilder import PathBuilder


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
    @staticmethod
    def encode(dataset: Dataset, params: EncoderParams) -> Dataset:
        cache_key = CacheHandler.generate_cache_key(Word2VecEncoder._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: Word2VecEncoder._encode_new_dataset(dataset, params))

        return encoded_dataset

    @staticmethod
    def _prepare_caching_params(dataset: Dataset, params: EncoderParams):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
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
        repertoires = np.zeros(shape=[dataset.get_repertoire_count(), vectors.vector_size])
        for (index, repertoire) in enumerate(encoded_dataset.get_data()):
            repertoires[index] = Word2VecEncoder._encode_repertoire(repertoire, vectors, params)

        labels = Word2VecEncoder._encode_labels(dataset, params)

        scaled_repertoires = Word2VecEncoder._scale_encoding(repertoires, params)

        encoded_dataset.params = dataset.params
        encoded_dataset.set_filenames(dataset.get_filenames())

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
    def _scale_encoding(repertoires: np.ndarray, params: EncoderParams):

        scaler_path = params["result_path"]
        scaler_file = scaler_path + FilenameHandler.get_filename(StandardScaler.__name__, "pickle")

        if os.path.isfile(scaler_file):
            with open(scaler_file, 'rb') as file:
                scaler = pickle.load(file)
                scaled_repertoires = scaler.transform(repertoires)
        else:
            scaler = StandardScaler()
            scaled_repertoires = scaler.fit_transform(repertoires)

            PathBuilder.build(scaler_path)

            with open(scaler_file, 'wb') as file:
                pickle.dump(scaler, file)

        return sparse.csc_matrix(scaled_repertoires)

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

