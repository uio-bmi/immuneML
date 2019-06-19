import math
import os
import pickle
from collections import Counter
from multiprocessing.pool import Pool

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.analysis.AxisType import AxisType
from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.ReadsType import ReadsType
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
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER, # continuous k-mers, gapped, IMGT-annotated or not
            "k": 3,                                                             # k-mer length
            ...
        },
        "batch_size": 1,
        "learn_model": True, # true for training set and false for test set
        "result_path": "../",
        "label_configuration": LabelConfiguration(), # labels should be set before encodings is invoked,
        "model_path": "../",
        "scaler_path": "../",
        "vectorizer_path": None
    }

    Parallelization is supported based on the value in the batch_size parameter. The same number of processes will be
    created as the batch_size parameter.
    """
    @staticmethod
    def encode(dataset: Dataset, params: EncoderParams) -> Dataset:

        filepath = params["result_path"] + "/" + params["filename"]

        if os.path.isfile(filepath):
            encoded_dataset = PickleLoader.load(filepath)
        else:
            encoded_dataset = KmerFrequencyEncoder._encode_new_dataset(dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: EncoderParams) -> Dataset:

        encoded_repertoire_list, repertoire_names, encoded_labels, feature_annotation_names = KmerFrequencyEncoder._encode_repertoires(dataset, params)
        vectorized_repertoires, feature_names = KmerFrequencyEncoder._vectorize_encoded_repertoires(repertoires=encoded_repertoire_list, params=params)
        normalized_repertoires = DataSummarizer.normalize_matrix(vectorized_repertoires, params["model"]["normalization_type"], AxisType.REPERTOIRES)
        feature_annotations = KmerFrequencyEncoder._get_feature_annotations(feature_names, feature_annotation_names)

        encoded_data = EncodedData(repertoires=normalized_repertoires,
                                   labels=encoded_labels,
                                   feature_names=feature_names,
                                   repertoire_ids=repertoire_names,
                                   feature_annotations=feature_annotations)

        encoded_dataset = Dataset(filenames=dataset.get_filenames(),
                                  encoded_data=encoded_data,
                                  params=dataset.params,
                                  metadata_file=dataset.metadata_file)

        KmerFrequencyEncoder.store(encoded_dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: Dataset, params: EncoderParams):

        arguments = [(filename, dataset, params) for filename in dataset.get_filenames()]

        print(arguments)
        print(len(arguments))

        print(params["batch_size"])
        print(math.floor(len(dataset.get_filenames())/params["batch_size"]) + 1)

        with Pool(params["batch_size"]) as pool:
            repertoires = pool.starmap(KmerFrequencyEncoder._encode_repertoire, arguments, chunksize=math.floor(len(dataset.get_filenames())/params["batch_size"]) + 1)

        encoded_repertoire_list, repertoire_names, labels, feature_annotation_names = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]}

        feature_annotation_names = feature_annotation_names[0]

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels, feature_annotation_names

    @staticmethod
    def _vectorize_encoded_repertoires(repertoires: list, params: EncoderParams):

        filename= params["result_path"] + FilenameHandler.get_filename(DictVectorizer.__name__, "pickle")

        if params["learn_model"]:
            vectorizer = DictVectorizer(sparse=True, dtype=float)
            vectorized_repertoires = vectorizer.fit_transform(repertoires)
            PathBuilder.build(params["result_path"])
            with open(filename, 'wb') as file:
                pickle.dump(vectorizer, file)
        else:
            with open(filename, 'rb') as file:
                vectorizer = pickle.load(file)
            vectorized_repertoires = vectorizer.transform(repertoires)

        return vectorized_repertoires, vectorizer.get_feature_names()

    @staticmethod
    def _get_feature_annotations(feature_names, feature_annotation_names):
        feature_annotations = pd.DataFrame({"feature": feature_names})
        feature_annotations[feature_annotation_names] = feature_annotations['feature'].str.split(Constants.FEATURE_DELIMITER, expand=True)
        return feature_annotations

    @staticmethod
    def _encode_repertoire(filename: str, dataset: Dataset, params: EncoderParams):
        repertoire = dataset.get_repertoire(filename=filename)
        counts = Counter()
        sequence_encoder = KmerFrequencyEncoder._prepare_sequence_encoder(params)
        feature_names = sequence_encoder.get_feature_names(params)
        for sequence in repertoire.sequences:
            features = sequence_encoder.encode_sequence(sequence, params)
            if features is not None:
                for i in features:
                    if params["model"].get('reads') == ReadsType.UNIQUE:
                        counts[i] += 1
                    elif params["model"].get('reads') == ReadsType.ALL:
                        counts[i] += sequence.metadata.count

        label_config = params["label_configuration"]
        labels = dict()

        for label_name in label_config.get_labels_by_name():
            label = repertoire.metadata.custom_params[label_name]
            labels[label_name] = label

        # TODO: refactor this not to return 4 values but e.g. a dict or split into different functions?
        return counts, repertoire.identifier, labels, feature_names

    @staticmethod
    def _prepare_sequence_encoder(params: EncoderParams):
        class_name = params["model"]["sequence_encoding"].value
        sequence_encoder = ReflectionHandler.get_class_by_name(class_name, "encodings")
        return sequence_encoder

    @staticmethod
    def store(encoded_dataset: Dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
