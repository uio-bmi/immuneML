import abc
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
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
        "RepertoireDataset": "KmerFreqRepertoireEncoder"
    }

    @staticmethod
    def create_encoder(dataset=None):
        return ReflectionHandler.get_class_by_name(KmerFrequencyEncoder.dataset_mapping[dataset.__class__.__name__], "kmer_frequency/")()

    def encode(self, dataset, params: EncoderParams):

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("encoding", KmerFrequencyEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("step", step),
                ("encoding_params", tuple([(key, params["model"][key]) for key in params["model"].keys()])), )

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
        class_name = params["model"]["sequence_encoding"].value
        sequence_encoder = ReflectionHandler.get_class_by_name(class_name, "encodings")
        return sequence_encoder

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
