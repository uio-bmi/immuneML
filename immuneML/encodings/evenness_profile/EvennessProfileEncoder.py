import abc

import numpy as np
import pandas as pd
from scipy import sparse

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.ReflectionHandler import ReflectionHandler


class EvennessProfileEncoder(DatasetEncoder):
    """
    The EvennessProfileEncoder class encodes a repertoire based on the clonal frequency distribution. The evenness for
    a given repertoire is defined as follows:

    .. math::

        ^{\\alpha} \\mathrm{E}(\\mathrm{f})=\\frac{\\left(\\sum_{\\mathrm{i}=1}^{\\mathrm{n}} \\mathrm{f}_{\\mathrm{i}}^{\\alpha}\\right)^{\\frac{1}{1-\\alpha}}}{\\mathrm{n}}

    That is, it is the exponential of Renyi entropy at a given alpha divided by the species richness, or number of unique
    sequences.

    See Greiff and colleagues' publication "A bioinformatic framework for immune repertoire diversity profiling enables
    detection of immunological status" in Genome Medicine 2015 for more details.

    Arguments:

        min_alpha (float): minimum alpha value to use

        max_alpha (float): maximum alpha value to use

        dimension (int): dimension of output evenness profile vector, or the number of alpha values to linearly space
        between min_alpha and max_alpha

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_evenness_profile:
                EvennessProfile:
                    min_alpha: 0
                    max_alpha: 10
                    dimension: 51


    """

    STEP_ENCODED = "encoded"
    STEP_VECTORIZED = "vectorized"

    dataset_mapping = {
        "RepertoireDataset": "EvennessProfileRepertoireEncoder",
    }

    def __init__(self, min_alpha: float, max_alpha: float, dimension: int, name: str = None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.dimension = dimension
        self.name = name

    @staticmethod
    def _prepare_parameters(min_alpha: float, max_alpha: float, dimension: int, name: str = None):

        return {
            "min_alpha": min_alpha,
            "max_alpha": max_alpha,
            "dimension": dimension,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_params = EvennessProfileEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(EvennessProfileEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "evenness_profile/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(EvennessProfileEncoder.__name__, dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):

        encoded_dataset = CacheHandler.memo_by_params(self._prepare_caching_params(dataset, params),
                                                      lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams, step: str = ""):
        return (("example_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file if hasattr(dataset, "metadata_file") else None),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", EvennessProfileEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", step),
                ("encoding_params", tuple(vars(self).items())))

    def _encode_data(self, dataset, params: EncoderParams) -> EncodedData:

        encoded_example_list, example_ids, encoded_labels = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, EvennessProfileEncoder.STEP_ENCODED),
            lambda: self._encode_examples(dataset, params))

        vectorized_examples = CacheHandler.memo_by_params(
            self._prepare_caching_params(dataset, params, EvennessProfileEncoder.STEP_VECTORIZED),
            lambda: self._vectorize_encoded(examples=encoded_example_list))

        feature_names = list(range(self.dimension))

        feature_annotations = pd.DataFrame({"feature": feature_names})

        encoded_data = EncodedData(examples=vectorized_examples,
                                   labels=encoded_labels,
                                   feature_names=feature_names,
                                   example_ids=example_ids,
                                   feature_annotations=feature_annotations,
                                   encoding=EvennessProfileEncoder.__name__)

        return encoded_data

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    @abc.abstractmethod
    def _encode_examples(self, dataset, params: EncoderParams):
        pass

    def _vectorize_encoded(self, examples: list):

        vectorized_examples = sparse.csr_matrix(np.vstack(examples))

        return vectorized_examples
