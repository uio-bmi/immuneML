import abc
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.caching.CacheHandler import CacheHandler
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.preprocessing.FeatureScaler import FeatureScaler
from source.environment.Constants import Constants
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler


class KmerFrequencyEncoder(DatasetEncoder):
    """
    Encodes the dataset based on frequency of each unique feature that results from the union of
    features across sequences in the dataset, based on one of the strategies of class
    SequenceEncodingStrategy.

    Configuration parameters are an instance of EncoderParams class:
    {
        "model": {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,         # relative frequencies of k-mers or L2
            "reads": ReadsType.UNIQUE,                                          # unique or all
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER,          # how to build k-mers
            "k": 3,                                                             # k-mer length
            ...
        },
        "batch_size": 1,
        "learn_model": True, # true for training set and false for test set
        "result_path": "../",
        "label_configuration": LabelConfiguration(), # labels should be set before encodings is invoked
    }

    Parallelization is supported based on the value in the batch_size parameter. The same number of processes will be
    created as the batch_size parameter.
    """

    STEP_ENCODED = "encoded"
    STEP_VECTORIZED = "vectorized"
    STEP_NORMALIZED = "normalized"

    dataset_mapping = {
        "RepertoireDataset": "KmerFreqRepertoireEncoder",
        "SequenceDataset": "KmerFreqSequenceEncoder",
        "ReceptorDataset": "KmerFreqReceptorEncoder"
    }

    def __init__(self, normalization_type: NormalizationType, reads: ReadsType, sequence_encoding: SequenceEncodingType, k: int = 0,
                 k_left: int = 0, k_right: int = 0, min_gap: int = 0, max_gap: int = 0, metadata_fields_to_include: list = None):
        self.normalization_type = normalization_type
        self.reads = reads
        self.sequence_encoding = sequence_encoding
        self.k = k
        self.k_left = k_left
        self.k_right = k_right
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.metadata_fields_to_include = metadata_fields_to_include if metadata_fields_to_include is not None else []

    @staticmethod
    def create_encoder(dataset=None, params: dict = None):
        try:
            encoder = ReflectionHandler.get_class_by_name(KmerFrequencyEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "kmer_frequency/")(**params if params is not None else {})
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(KmerFrequencyEncoder.__name__, dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file if hasattr(dataset, "metadata_file") else None),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("encoding", KmerFrequencyEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("step", step),
                ("encoding_params", tuple(vars(self))))

    def _encode_data(self, dataset, params: EncoderParams) -> EncodedData:
        encoded_example_list, example_ids, encoded_labels, feature_annotation_names = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_ENCODED),
            lambda: self._encode_examples(dataset, params))

        vectorized_examples, feature_names = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_VECTORIZED),
            lambda: self._vectorize_encoded(examples=encoded_example_list, params=params))

        normalized_examples = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_NORMALIZED),
            lambda: FeatureScaler.normalize(params["result_path"] + "normalizer.pkl",
                                            vectorized_examples,
                                            self.normalization_type))

        feature_annotations = self._get_feature_annotations(feature_names, feature_annotation_names)

        encoded_data = EncodedData(examples=normalized_examples,
                                   labels=encoded_labels,
                                   feature_names=feature_names,
                                   example_ids=example_ids,
                                   feature_annotations=feature_annotations,
                                   encoding=KmerFrequencyEncoder.__name__)

        return encoded_data

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    @abc.abstractmethod
    def _encode_examples(self, dataset, params: EncoderParams):
        pass

    def _vectorize_encoded(self, examples: list, params: EncoderParams):

        filename = params["result_path"] + FilenameHandler.get_filename(DictVectorizer.__name__, "pickle")

        if params["learn_model"]:
            vectorizer = DictVectorizer(sparse=True, dtype=float)
            vectorized_examples = vectorizer.fit_transform(examples)
            PathBuilder.build(params["result_path"])
            with open(filename, 'wb') as file:
                pickle.dump(vectorizer, file)
        else:
            with open(filename, 'rb') as file:
                vectorizer = pickle.load(file)
            vectorized_examples = vectorizer.transform(examples)

        return vectorized_examples, vectorizer.get_feature_names()

    def _get_feature_annotations(self, feature_names, feature_annotation_names):
        feature_annotations = pd.DataFrame({"feature": feature_names})
        feature_annotations[feature_annotation_names] = feature_annotations['feature'].str.split(Constants.FEATURE_DELIMITER, expand=True)
        return feature_annotations

    def _prepare_sequence_encoder(self, params: EncoderParams):
        class_name = self.sequence_encoding.value
        sequence_encoder = ReflectionHandler.get_class_by_name(class_name, "encodings")
        return sequence_encoder

    def _encode_sequence(self, sequence: ReceptorSequence, params: EncoderParams, sequence_encoder, counts):
        params["model"] = vars(self)
        features = sequence_encoder.encode_sequence(sequence, params)
        if features is not None:
            for i in features:
                if self.reads == ReadsType.UNIQUE:
                    counts[i] += 1
                elif self.reads == ReadsType.ALL:
                    counts[i] += sequence.metadata.count
        return counts

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
