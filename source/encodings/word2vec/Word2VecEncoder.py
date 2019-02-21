# quality: gold

import os
import pickle
from copy import copy

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from source.IO.PickleExporter import PickleExporter
from source.IO.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireGenerator import RepertoireGenerator
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.KmerHelper import KmerHelper
from source.util.NameBuilder import NameBuilder
from source.util.PathBuilder import PathBuilder


class Word2VecEncoder(DatasetEncoder):
    """
    Encodes the dataset using Word2Vec model.
    It relies on gensim's implementation of Word2Vec and KmerHelper for k-mer extraction.

    Configuration parameters example:
    {
        "model": {
            "k": 3,
            "model_creator": ModelType.SEQUENCE,
            "size": 16
        },
        "batch_size": 1,
        "learn_model": True, # true for training set and false for test set
        "result_path": "../",
        "label_configuration": LabelConfiguration(), # labels should be set before encodings is invoked,
        "model_path": "../",
        "scaler_path": "../"
    }

    NB: In order to use the workers properly and be able to parallelize the training process,
    it is necessary that Cython is installed on the machine.
    """
    @staticmethod
    def encode(dataset: Dataset, params: dict) -> Dataset:

        Word2VecEncoder.validate_configuration(params)

        filepath = params["result_path"] + "encoded_dataset.pkl"

        if os.path.isfile(filepath):
            encoded_dataset = PickleLoader.load(filepath)
        else:
            encoded_dataset = Word2VecEncoder._encode_new_dataset(dataset, params)

        return encoded_dataset

    @staticmethod
    def validate_configuration(params: dict):
        assert "model" in params, "Word2VecEncoder: the model dict is missing in the parameters."
        assert "k" in params["model"] and isinstance(params["model"]["k"], int), "Word2VecEncoder: the k param is not set in the parameters of the model."
        assert "model_creator" in params["model"] and isinstance(params["model"]["model_creator"], ModelType), "Word2VecEncoder: the model_creator param is not set properly in the parameters of the model."
        assert "size" in params["model"], "Word2VecEncoder: the size param is not set properly in the parameters of the model."
        assert "batch_size" in params and isinstance(params["batch_size"], int), "Word2VecEncoder: the batch_size param is not set in the parameters."
        assert "learn_model" in params and isinstance(params["learn_model"], bool), "Word2VecEncoder: the learn_model param is not set in the parameters."
        assert "result_path" in params, "Word2VecEncoder: the result_path param is not set in the parameters."
        assert "label_configuration" in params and isinstance(params["label_configuration"], LabelConfiguration), "Word2VecEncoder: the label_configuration param is not properly set in the parameters."
        assert "model_path" in params, "Word2VecEncoder: the model_path param is not set in the parameters."
        assert "scaler_path" in params, "Word2VecEncoder: the scaler_path param is not set in the parameters."

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: dict) -> Dataset:
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
    def _encode_repertoire(repertoire: Repertoire, vectors, params):
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
    def _encode_labels(dataset: Dataset, params: dict):

        label_config = params["label_configuration"]
        labels = {name: [] for name in label_config.get_labels_by_name()}

        for repertoire in dataset.get_data(params["batch_size"]):

            sample = repertoire.metadata.sample

            for label_name in label_config.get_labels_by_name():
                label = sample.custom_params[label_name]

                # binarizer = label_config.get_label_binarizer(label_name)
                # label = binarizer.transform([label])

                # TODO: binarization removed from encodings because ML methods require different inputs:
                #   this will be part of the ML method itself; but remove this comment when this is checked

                labels[label_name].append(label)

        return np.array([labels[name] for name in labels.keys()])

    @staticmethod
    def _encode_by_model(dataset, vectors, params) -> Dataset:

        encoded_dataset = Dataset()
        data = RepertoireGenerator.build_generator(dataset.filenames, params["batch_size"])
        repertoires = np.zeros(shape=[dataset.get_repertoire_count(), vectors.vector_size])
        for (index, repertoire) in enumerate(data):
            repertoires[index] = Word2VecEncoder._encode_repertoire(repertoire, vectors, params)

        labels = Word2VecEncoder._encode_labels(dataset, params)

        scaled_repertoires = Word2VecEncoder._scale_encoding(repertoires, params)

        encoded_dataset.params = dataset.params

        encoded_dataset.add_encoded_data({
            "repertoires": scaled_repertoires,
            "labels": labels,
            "feature_names": [],
            "label_names": params["label_configuration"].get_labels_by_name()
        })

        return encoded_dataset

    @staticmethod
    def _scale_encoding(repertoires: np.ndarray, params):

        scaler_path = params["scaler_path"]
        scaler_file = scaler_path + "scaler.pkl"

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
    def _load_model(params):
        model_path = Word2VecEncoder._create_model_path(params)
        model = Word2Vec.load(model_path)
        return model

    @staticmethod
    def _create_model(dataset, params):

        if params["model"]["model_creator"] == ModelType.SEQUENCE:
            model_creator = SequenceModelCreator()
        else:
            model_creator = KmerPairModelCreator()

        model = model_creator.create_model(dataset=dataset,
                                           params=params,
                                           model_path=Word2VecEncoder._create_model_path(params))

        return model

    @staticmethod
    def store(encoded_dataset: Dataset, params: dict):
        PickleExporter.export(encoded_dataset, params["result_path"], "encoded_dataset.pkl")

    @staticmethod
    def _exists_model(params) -> bool:
        return os.path.isfile(Word2VecEncoder._create_model_path(params))

    @staticmethod
    def _create_model_path(params):
        return params["model_path"] + "W2V_" + NameBuilder.build_name_from_dict(params["model"]) + ".model"

