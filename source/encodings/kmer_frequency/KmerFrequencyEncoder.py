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
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.GappedKmerSequenceEncoder import GappedKmerSequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.IMGTGappedKmerEncoder import IMGTGappedKmerEncoder
from source.encodings.kmer_frequency.sequence_encoding.IMGTKmerSequenceEncoder import IMGTKmerSequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class KmerFrequencyEncoder(DatasetEncoder):

    @staticmethod
    def validate_configuration(params: dict):
        assert "result_path" in params, "KmerFrequencyEncoder: the result_path parameter is not set in params."
        assert "label_configuration" in params and isinstance(params["label_configuration"], LabelConfiguration), "KmerFrequencyEncoder: the label_configuration parameter is not properly set in params."
        assert "batch_size" in params and isinstance(params["batch_size"], int), "KmerFrequencyEncoder: the batch_size param is not set in the parameters."
        assert "learn_model" in params and isinstance(params["learn_model"], bool), "KmerFrequencyEncoder: the learn_model param is not set in the parameters."
        assert "vectorizer_path" in params, "KmerFrequencyEncoder: the vectorizer_path param is not set in the parameters."
        assert "normalization_type" in params and isinstance(params["normalization_type"], NormalizationType), "KmerFrequencyEncoder: the normalization_type param is not properly set in the parameters."
        assert "reads" in params and isinstance(params["reads"], ReadsType), "KmerFrequencyEncoder: the reads param is not properly set in the parameters."
        assert "sequence_encoding_strategy" in params and isinstance(params["sequence_encoding_strategy"], SequenceEncodingType), "KmerFrequencyEncoder: the sequence_encoding_strategy param is not properly set in the parameters."

    @staticmethod
    def encode(dataset: Dataset, params: dict) -> Dataset:

        KmerFrequencyEncoder.validate_configuration(params)

        filepath = params["result_path"] + "encoded_dataset.pkl"

        if os.path.isfile(filepath):
            encoded_dataset = PickleLoader.load(filepath)
        else:
            encoded_dataset = KmerFrequencyEncoder._encode_new_dataset(dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: dict) -> Dataset:

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
                                  encoded_data=encoded_dataset,
                                  params=dataset.params)

        KmerFrequencyEncoder.store(encoded_dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: Dataset, params: dict):
        encoded_repertoire_list = []
        for repertoire in dataset.get_data(params["batch_size"]):
            encoded_repertoire_list.append(KmerFrequencyEncoder._encode_repertoire(repertoire, params))
        return encoded_repertoire_list

    @staticmethod
    def _vectorize_encoded_repertoires(repertoires: list, params: dict):
        if params["learn_model"]:
            vectorizer = DictVectorizer(sparse=True, dtype=float)
            vectorized_repertoires = vectorizer.fit_transform(repertoires)

            PathBuilder.build(params["vectorizer_path"])
            with open(params["vectorizer_path"] + "DictVectorizer.pkl", 'wb') as file:
                pickle.dump(vectorizer, file)
        else:
            with open(params["vectorizer_path"] + "DictVectorizer.pkl", 'rb') as file:
                vectorizer = pickle.load(file)
            vectorized_repertoires = vectorizer.transform(repertoires)

        return vectorized_repertoires, vectorizer.get_feature_names()

    @staticmethod
    def _normalize_repertoires(repertoires, params: dict):
        normalized_repertoires = repertoires
        if params.get('normalization_type') == NormalizationType.RELATIVE_FREQUENCY:
            normalized_repertoires = sparse.diags(1 / repertoires.sum(axis=1).A.ravel()) @ repertoires
        elif params.get('normalization_type') == NormalizationType.L2:
            normalized_repertoires = normalize(repertoires)
        return normalized_repertoires

    @staticmethod
    def _encode_labels(dataset: Dataset, params: dict):

        label_config = params["label_configuration"]
        labels = {name: [] for name in label_config.get_labels_by_name()}

        for repertoire in dataset.get_data(params["batch_size"]):

            sample = repertoire.metadata.sample

            for label_name in label_config.get_labels_by_name():
                label = sample.custom_params[label_name]

                #binarizer = label_config.get_label_binarizer(label_name)
                #label = binarizer.transform([label])

                # TODO: binarization removed from encodings because ML methods require different inputs:
                #   this will be part of the ML method itself; but remove this comment when this is checked

                labels[label_name].append(label)

        return np.array([labels[name] for name in labels.keys()])

    @staticmethod
    def store(encoded_dataset: Dataset, params: dict):
        PickleExporter.export(encoded_dataset, params["result_path"], "encoded_dataset.pkl")

    @staticmethod
    def _encode_repertoire(repertoire: Repertoire, params: dict):
        counts = Counter()
        for sequence in repertoire.sequences:
            features = KmerFrequencyEncoder._encode_sequence(sequence, params)
            for i in features:
                if params.get('reads') == ReadsType.UNIQUE:
                    counts[i] += 1
                elif params.get('reads') == ReadsType.ALL:
                    counts[i] += sequence.count
        return counts

    @staticmethod
    def _encode_sequence(sequence: ReceptorSequence, params: dict):
        sequence_encoder = KmerFrequencyEncoder._prepare_sequence_encoder(params)
        encoded_sequence = sequence_encoder.encode_sequence(sequence=sequence, params=params)
        return encoded_sequence

    @staticmethod
    def _prepare_sequence_encoder(params: dict):
        if params["sequence_encoding_strategy"] == SequenceEncodingType.CONTINUOUS_KMER:
            sequence_encoder = KmerSequenceEncoder()
        elif params["sequence_encoding_strategy"] == SequenceEncodingType.GAPPED_KMER:
            sequence_encoder = GappedKmerSequenceEncoder()
        elif params["sequence_encoding_strategy"] == SequenceEncodingType.IMGT_CONTINUOUS_KMER:
            sequence_encoder = IMGTKmerSequenceEncoder()
        elif params["sequence_encoding_strategy"] == SequenceEncodingType.IMGT_GAPPPED_KMER:
            sequence_encoder = IMGTGappedKmerEncoder()
        else:
            sequence_encoder = IdentitySequenceEncoder()

        return sequence_encoder
