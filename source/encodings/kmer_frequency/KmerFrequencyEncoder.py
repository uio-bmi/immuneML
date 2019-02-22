import copy
import os
import pickle
from collections import Counter

from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
import numpy as np

from source.IO.PickleExporter import PickleExporter
from source.IO.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.DatasetParams import DatasetParams
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder


class KmerFrequencyEncoder(DatasetEncoder):

    @staticmethod
    def encode(dataset: Dataset, params: EncoderParams) -> Dataset:

        filepath = params["result_path"] + FilenameHandler.get_dataset_name(KmerFrequencyEncoder.__name__)

        if os.path.isfile(filepath):
            encoded_dataset = PickleLoader.load(filepath)
        else:
            encoded_dataset = KmerFrequencyEncoder._encode_new_dataset(dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: EncoderParams) -> Dataset:

        # TODO: add parallelism to this step: k-mer frequencies in reps can be calculated in parallel
        #  even if other parts cannot be parallel 
        encoded_repertoire_list = KmerFrequencyEncoder._encode_repertoires(dataset, params)

        vectorized_repertoires, feature_names = KmerFrequencyEncoder._vectorize_encoded_repertoires(repertoires=encoded_repertoire_list, params=params)
        normalized_repertoires = KmerFrequencyEncoder._normalize_repertoires(repertoires=vectorized_repertoires, params=params)
        encoded_labels = KmerFrequencyEncoder._encode_labels(dataset, params)

        encoded_dataset = {'repertoires': normalized_repertoires.toarray(),
                           'labels': encoded_labels,
                           'label_names': params["label_configuration"].get_labels_by_name(),
                           'feature_names': feature_names}

        encoded_dataset = Dataset(filenames=dataset.filenames,
                                  encoded_data=encoded_dataset)

        KmerFrequencyEncoder.store(encoded_dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: Dataset, params: EncoderParams):
        encoded_repertoire_list = []
        for repertoire in dataset.get_data(params["batch_size"]):
            encoded_repertoire_list.append(KmerFrequencyEncoder._encode_repertoire(repertoire, params))
        return encoded_repertoire_list

    @staticmethod
    def _vectorize_encoded_repertoires(repertoires: list, params: EncoderParams):

        filepath = params["vectorizer_path"] + FilenameHandler.get_filename(DictVectorizer.__name__, "pickle")

        if params["learn_model"]:
            vectorizer = DictVectorizer(sparse=True, dtype=float)
            vectorized_repertoires = vectorizer.fit_transform(repertoires)
            PathBuilder.build(params["vectorizer_path"])
            with open(filepath, 'wb') as file:
                pickle.dump(vectorizer, file)
        else:
            with open(filepath, 'rb') as file:
                vectorizer = pickle.load(file)
            vectorized_repertoires = vectorizer.transform(repertoires)

        return vectorized_repertoires, vectorizer.get_feature_names()

    @staticmethod
    def _normalize_repertoires(repertoires, params: EncoderParams):
        normalized_repertoires = repertoires
        if params.get('normalization_type') == NormalizationType.RELATIVE_FREQUENCY:
            normalized_repertoires = sparse.diags(1 / repertoires.sum(axis=1).A.ravel()) @ repertoires
        elif params.get('normalization_type') == NormalizationType.L2:
            normalized_repertoires = normalize(repertoires)
        return normalized_repertoires

    @staticmethod
    def _encode_labels(dataset: Dataset, params: EncoderParams):

        label_config = params["label_configuration"]
        labels = {name: [] for name in label_config.get_labels_by_name()}

        for repertoire in dataset.get_data(params["batch_size"]):

            sample = repertoire.metadata.sample

            for label_name in label_config.get_labels_by_name():
                label = sample.custom_params[label_name]
                labels[label_name].append(label)

        return np.array([labels[name] for name in labels.keys()])

    @staticmethod
    def store(encoded_dataset: Dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"],
                              FilenameHandler.get_dataset_name(KmerFrequencyEncoder.__name__))

    @staticmethod
    def _encode_repertoire(repertoire: Repertoire, params: EncoderParams):
        counts = Counter()
        for sequence in repertoire.sequences:
            features = KmerFrequencyEncoder._encode_sequence(sequence, params)
            for i in features:
                if params["model"].get('reads') == ReadsType.UNIQUE:
                    counts[i] += 1
                elif params["model"].get('reads') == ReadsType.ALL:
                    counts[i] += sequence.count
        return counts

    @staticmethod
    def _encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        sequence_encoder = KmerFrequencyEncoder._prepare_sequence_encoder(params)
        encoded_sequence = sequence_encoder.encode_sequence(sequence=sequence, params=params)
        return encoded_sequence

    @staticmethod
    def _prepare_sequence_encoder(params: EncoderParams):
        from importlib import import_module
        module_path, _, class_name = params["model"]["sequence_encoding_strategy"].value.rpartition('.')
        mod = import_module(module_path)
        sequence_encoder = getattr(mod, class_name)()
        return sequence_encoder
