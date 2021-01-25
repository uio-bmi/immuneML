# quality: gold
import abc
import hashlib
from pathlib import Path
from typing import List

import pandas as pd
from gensim.models import Word2Vec

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.encodings.word2vec.model_creator.KmerPairModelCreator import KmerPairModelCreator
from immuneML.encodings.word2vec.model_creator.ModelType import ModelType
from immuneML.encodings.word2vec.model_creator.SequenceModelCreator import SequenceModelCreator
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


class Word2VecEncoder(DatasetEncoder):
    """

    Word2VecEncoder learns the vector representations of k-mers in the sequences in a repertoire from the
    context the k-mers appear in.
    It relies on gensim's implementation of Word2Vec and KmerHelper for k-mer extraction.


    Arguments:

        vector_size (int): The size of the vector to be learnt.

        model_type (:py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType`):  The context which will be
        used to infer the representation of the sequence.
        If :py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType.SEQUENCE` is used, the context of
        a k-mer is defined by the sequence it occurs in (e.g. if the sequence is CASTTY and k-mer is AST,
        then its context consists of k-mers CAS, STT, TTY)
        If :py:obj:`~immuneML.encodings.word2vec.model_creator.ModelType.ModelType.KMER_PAIR` is used, the context for
        the k-mer is defined as all the k-mers that within one edit distance (e.g. for k-mer CAS, the context
        includes CAA, CAC, CAD etc.).
        Valid values for this parameter are names of the ModelType enum.

        k (int): The length of the k-mers used for the encoding.


    YAML specification:

    .. highlight:: yaml
    .. code-block:: yaml

        encodings:
            my_w2v:
                Word2Vec:
                    vector_size: 16
                    k: 3
                    model_type: SEQUENCE

    """

    DESCRIPTION_REPERTOIRES = "repertoires"
    DESCRIPTION_LABELS = "labels"

    dataset_mapping = {
        "RepertoireDataset": "W2VRepertoireEncoder"
    }

    def __init__(self, vector_size: int, k: int, model_type: ModelType, name: str = None):
        self.vector_size = vector_size
        self.k = k
        self.model_type = model_type
        self.model_path = None
        self.name = name

    @staticmethod
    def _prepare_parameters(vector_size: int, k: int, model_type: str, name: str = None):
        location = "Word2VecEncoder"
        ParameterValidator.assert_type_and_value(vector_size, int, location, "vector_size", min_inclusive=1)
        ParameterValidator.assert_type_and_value(k, int, location, "k", min_inclusive=1)
        ParameterValidator.assert_in_valid_list(model_type.upper(), [item.name for item in ModelType], location, "model_type")
        return {"vector_size": vector_size, "k": k, "model_type": ModelType[model_type.upper()], "name": name}

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_params = Word2VecEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(
                Word2VecEncoder.dataset_mapping[dataset.__class__.__name__], "word2vec/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(Word2VecEncoder.__name__,
                                                                                dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):
        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, vectors=None, description: str = ""):
        return (("dataset_id", dataset.identifier),
                ("dataset_filenames", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("vectors", hashlib.sha256(str(vectors).encode("utf-8")).hexdigest()),
                ("description", description),
                ("encoding", Word2VecEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", tuple([(key, params.model[key]) for key in params.model.keys()])), )

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        if params.learn_model is True and not self._exists_model(params):
            model = self._create_model(dataset=dataset, params=params)
        else:
            model = self._load_model(params)

        vectors = model.wv
        del model

        encoded_dataset = self._encode_by_model(dataset, params, vectors)

        return encoded_dataset

    @abc.abstractmethod
    def _encode_labels(self, dataset, params: EncoderParams):
        pass

    def _encode_by_model(self, dataset, params: EncoderParams, vectors):
        examples = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params, vectors,
                                                                            Word2VecEncoder.DESCRIPTION_REPERTOIRES),
                                               lambda: self._encode_examples(dataset, vectors, params))

        if params.encode_labels:
            labels = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params, vectors, Word2VecEncoder.DESCRIPTION_LABELS),
                                                 lambda: self._encode_labels(dataset, params))
        else:
            labels = None

        scaler_filename = params.result_path / FilenameHandler.get_filename("standard_scaling", "pkl")
        scaled_examples = FeatureScaler.standard_scale(scaler_filename, examples)

        encoded_dataset = self._build_encoded_dataset(dataset, scaled_examples, labels, params)
        return encoded_dataset

    def _build_encoded_dataset(self, dataset: Dataset, scaled_examples, labels, params: EncoderParams):

        encoded_dataset = dataset.clone()

        label_names = params.label_config.get_labels_by_name()
        feature_names = [str(i) for i in range(scaled_examples.shape[1])]
        feature_annotations = pd.DataFrame({"feature": feature_names})

        encoded_data = EncodedData(examples=scaled_examples,
                                   labels={label: labels[i] for i, label in enumerate(label_names)} if labels is not None else None,
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
        model = Word2Vec.load(str(model_path))
        return model

    def _create_model(self, dataset, params: EncoderParams):

        if self.model_path is None:
            self.model_path = self._create_model_path(params)

        if self.model_type == ModelType.SEQUENCE:
            model_creator = SequenceModelCreator()
        else:
            model_creator = KmerPairModelCreator()

        model = model_creator.create_model(dataset=dataset,
                                           k=self.k,
                                           vector_size=self.vector_size,
                                           batch_size=params.pool_size,
                                           model_path=self.model_path)

        return model

    def _exists_model(self, params: EncoderParams) -> bool:
        return self._create_model_path(params).is_file()

    def _create_model_path(self, params: EncoderParams) -> Path:
        if self.model_path is None:
            return params.result_path / "W2V.model"
        else:
            return self.model_path

    def get_additional_files(self) -> List[str]:
        return [self.model_path]

    @staticmethod
    def export_encoder(path: Path, encoder) -> str:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder = DatasetEncoder.load_attribute(encoder, encoder_file, "model_path")
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(Word2VecEncoder.__doc__)

        valid_values = str([model_type.name for model_type in ModelType])[1:-1].replace("'", "`")
        mapping = {
            "Valid values for this parameter are names of the ModelType enum.": f"Valid values are {valid_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
