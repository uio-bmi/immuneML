import abc
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.receptor_sequence import ReceptorSequence
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.environment.Constants import Constants
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


class KmerFrequencyEncoder(DatasetEncoder):
    """
    The KmerFrequencyEncoder class encodes a repertoire, sequence or receptor by frequencies of k-mers it contains.
    A k-mer is a sequence of letters of length k into which an immune receptor sequence can be decomposed.
    K-mers can be defined in different ways, as determined by the sequence_encoding.


    Arguments:

        sequence_encoding (:py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType`):
        The type of k-mers that are used. The simplest sequence_encoding is :py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.CONTINUOUS_KMER`,
        which simply uses contiguous subsequences of length k to represent the k-mers.
        When gapped k-mers are used (:py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`,
        :py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.GAPPED_KMER`), the k-mers may contain
        gaps with a size between min_gap and max_gap, and the k-mer length is defined as a combination of k_left and k_right.
        When IMGT k-mers are used (:py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_CONTINUOUS_KMER`,
        :py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IMGT_GAPPED_KMER`), IMGT positional information is
        taken into account (i.e. the same sequence in a different position is considered to be a different k-mer).
        When the identity representation is used (:py:mod:`immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType.IDENTITY`),
        the k-mers just correspond to the original sequences.

        normalization_type (:py:mod:`immuneML.analysis.data_manipulation.NormalizationType`): The way in which the
        k-mer frequencies should be normalized. The default value for normalization_type is l2.

        reads (:py:mod:`immuneML.encodings.kmer_frequency.ReadsType`): Reads type signify whether the counts of the sequences
        in the repertoire will be taken into account. If :py:mod:`immuneML.encodings.kmer_frequency.ReadsType.UNIQUE`,
        only unique sequences (clonotypes) are encoded, and if :py:mod:`immuneML.encodings.kmer_frequency.ReadsType.ALL`,
        the sequence 'count' value is taken into account when determining the k-mer frequency.
        The default value for reads is unique.

        k (int): Length of the k-mer (number of amino acids) when ungapped k-mers are used. The default value for k is 3.

        k_left (int): When gapped k-mers are used, k_left indicates the length of the k-mer left of the gap. The default value for k_left is 1.

        k_right (int): Same as k_left, but k_right determines the length of the k-mer right of the gap. The default value for k_right is 1.

        min_gap (int): Minimum gap size when gapped k-mers are used. The default value for min_gap is 0.

        max_gap: (int): Maximum gap size when gapped k-mers are used. The default value for max_gap is 0.

        scale_to_unit_variance (bool): whether to scale the design matrix after normalization to have unit variance per feature. Setting this argument
        to True might improve the subsequent classifier's performance depending on the type of the classifier. The default value for scale_to_unit_variance is true.

        scale_to_zero_mean (bool): whether to scale the design matrix after normalization to have zero mean per feature. Setting this argument to True
        might improve the subsequent classifier's performance depending on the type of the classifier. However, if the original design matrix was
        sparse, setting this argument to True will destroy the sparsity and will increase the memory consumption. The default value for scale_to_zero_mean is false.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_continuous_kmer:
                KmerFrequency:
                    normalization_type: RELATIVE_FREQUENCY
                    reads: UNIQUE
                    sequence_encoding: CONTINUOUS_KMER
                    k: 3
                    scale_to_unit_variance: True
                    scale_to_zero_mean: True
            my_gapped_kmer:
                KmerFrequency:
                    normalization_type: RELATIVE_FREQUENCY
                    reads: UNIQUE
                    sequence_encoding: GAPPED_KMER
                    k_left: 2
                    k_right: 2
                    min_gap: 1
                    max_gap: 3
                    scale_to_unit_variance: True
                    scale_to_zero_mean: False

    """

    STEP_ENCODED = "encoded"
    STEP_VECTORIZED = "vectorized"
    STEP_NORMALIZED = "normalized"
    STEP_SCALED = "scaled"

    dataset_mapping = {
        "RepertoireDataset": "KmerFreqRepertoireEncoder",
        "SequenceDataset": "KmerFreqSequenceEncoder",
        "ReceptorDataset": "KmerFreqReceptorEncoder"
    }

    def __init__(self, normalization_type: NormalizationType, reads: ReadsType, sequence_encoding: SequenceEncodingType, k: int = 0,
                 k_left: int = 0, k_right: int = 0, min_gap: int = 0, max_gap: int = 0, metadata_fields_to_include: list = None,
                 name: str = None, scale_to_unit_variance: bool = False, scale_to_zero_mean: bool = False):
        self.normalization_type = normalization_type
        self.reads = reads
        self.sequence_encoding = sequence_encoding
        self.k = k
        self.k_left = k_left
        self.k_right = k_right
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.metadata_fields_to_include = metadata_fields_to_include if metadata_fields_to_include is not None else []
        self.name = name
        self.scale_to_unit_variance = scale_to_unit_variance
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scaler_path = None
        self.vectorizer_path = None

    @staticmethod
    def _prepare_parameters(normalization_type: str, reads: str, sequence_encoding: str, k: int = 0, k_left: int = 0,
                            k_right: int = 0, min_gap: int = 0, max_gap: int = 0, metadata_fields_to_include: list = None, name: str = None,
                            scale_to_unit_variance: bool = False, scale_to_zero_mean: bool = False):

        location = KmerFrequencyEncoder.__name__

        ParameterValidator.assert_in_valid_list(normalization_type.upper(), [item.name for item in NormalizationType], location, "normalization_type")
        ParameterValidator.assert_in_valid_list(reads.upper(), [item.name for item in ReadsType], location, "reads")
        ParameterValidator.assert_in_valid_list(sequence_encoding.upper(), [item.name for item in SequenceEncodingType], location, "sequence_encoding")
        ParameterValidator.assert_type_and_value(scale_to_zero_mean, bool, location, "scale_to_zero_mean")
        ParameterValidator.assert_type_and_value(scale_to_unit_variance, bool, location, "scale_to_unit_variance")

        vars_to_check = {"k": k, "k_left": k_left, "k_right": k_right, "min_gap": min_gap, "max_gap": max_gap}
        for param in vars_to_check.keys():
            ParameterValidator.assert_type_and_value(vars_to_check[param], int, location, param, min_inclusive=0)

        if "gap" in sequence_encoding.lower():
            assert k_left != 0 and k_right != 0, f"KmerFrequencyEncoder: sequence encoding {sequence_encoding} was chosen, but k_left " \
                                                 f"({k_left}) or k_right ({k_right}) have to be set and larger than 0."

        return {
            "normalization_type": NormalizationType[normalization_type.upper()],
            "reads": ReadsType[reads.upper()],
            "sequence_encoding": SequenceEncodingType[sequence_encoding.upper()],
            "name": name,
            "scale_to_zero_mean": scale_to_zero_mean, "scale_to_unit_variance": scale_to_unit_variance,
            **vars_to_check
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_params = KmerFrequencyEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(KmerFrequencyEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "kmer_frequency/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(KmerFrequencyEncoder.__name__, dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("dataset_identifier", dataset.identifier),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", KmerFrequencyEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", step),
                ("encoding_params", tuple(vars(self).items())))

    def _encode_data(self, dataset, params: EncoderParams) -> EncodedData:
        encoded_example_list, example_ids, encoded_labels, feature_annotation_names = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_ENCODED),
            lambda: self._encode_examples(dataset, params))

        vectorized_examples, feature_names = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_VECTORIZED),
            lambda: self._vectorize_encoded(examples=encoded_example_list, params=params))

        normalized_examples = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, KmerFrequencyEncoder.STEP_NORMALIZED),
            lambda: FeatureScaler.normalize(vectorized_examples, self.normalization_type))

        if self.scale_to_unit_variance:
            examples = self.scale_normalized(params, dataset, normalized_examples)
        else:
            examples = normalized_examples

        feature_annotations = self._get_feature_annotations(feature_names, feature_annotation_names)

        encoded_data = EncodedData(examples=examples,
                                   labels=encoded_labels,
                                   feature_names=feature_names,
                                   example_ids=example_ids,
                                   feature_annotations=feature_annotations,
                                   encoding=KmerFrequencyEncoder.__name__)

        return encoded_data

    def scale_normalized(self, params, dataset, normalized_examples):
        self.scaler_path = params.result_path / 'scaler.pickle' if self.scaler_path is None else self.scaler_path

        examples = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, step=KmerFrequencyEncoder.STEP_SCALED),
            lambda: FeatureScaler.standard_scale(self.scaler_path, normalized_examples, with_mean=self.scale_to_zero_mean))

        return examples

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    @abc.abstractmethod
    def _encode_examples(self, dataset, params: EncoderParams):
        pass

    def _vectorize_encoded(self, examples: list, params: EncoderParams):

        if self.vectorizer_path is None:
            self.vectorizer_path = params.result_path / FilenameHandler.get_filename(DictVectorizer.__name__, "pickle")

        if params.learn_model:
            vectorizer = DictVectorizer(sparse=True, dtype=float)
            vectorized_examples = vectorizer.fit_transform(examples)
            PathBuilder.build(params.result_path)
            with self.vectorizer_path.open('wb') as file:
                pickle.dump(vectorizer, file)
        else:
            with self.vectorizer_path.open('rb') as file:
                vectorizer = pickle.load(file)
            vectorized_examples = vectorizer.transform(examples)

        return vectorized_examples, vectorizer.get_feature_names()

    def _get_feature_annotations(self, feature_names, feature_annotation_names):
        feature_annotations = pd.DataFrame({"feature": feature_names})
        feature_annotations[feature_annotation_names] = feature_annotations['feature'].str.split(Constants.FEATURE_DELIMITER, expand=True)
        return feature_annotations

    def _prepare_sequence_encoder(self):
        class_name = self.sequence_encoding.value
        sequence_encoder = ReflectionHandler.get_class_by_name(class_name, "encodings/")
        return sequence_encoder

    def _encode_sequence(self, sequence: ReceptorSequence, params: EncoderParams, sequence_encoder, counts):
        params.model = vars(self)
        features = sequence_encoder.encode_sequence(sequence, params)
        if features is not None:
            for i in features:
                if self.reads == ReadsType.UNIQUE:
                    counts[i] += 1
                elif self.reads == ReadsType.ALL:
                    counts[i] += sequence.metadata.count
        return counts

    def get_additional_files(self) -> List[str]:
        files = []
        if self.scaler_path is not None and self.scaler_path.is_file():
            files.append(self.scaler_path)
        if self.vectorizer_path is not None and self.vectorizer_path.is_file():
            files.append(self.vectorizer_path)
        return files

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        for attribute in ['scaler_path', 'vectorizer_path']:
            encoder = DatasetEncoder.load_attribute(encoder, encoder_file, attribute)
        return encoder
