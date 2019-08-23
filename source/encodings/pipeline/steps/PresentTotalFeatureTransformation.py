import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import TransformerMixin

from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData


class PresentTotalFeatureTransformation(TransformerMixin):
    """
    Creates a data representation with only two features: the sum of the values of features meeting the criteria and
    the sum of the values for all features present in the dataset.
    If nonzero=True, then instead of summing the values in the dataset, it counts the number of nonzero values to
    compute both features.
    This is how Emerson et al., 2017 Nature Genetics represented their data after performing Fisher's exact test to filter sequences.
    This is a step in the PipelineEncoder.
    """
    def __init__(self, criteria, nonzero=True, result_path=None, filename=None):
        self.criteria = criteria
        self.nonzero = nonzero
        self.result_path = result_path
        self.filename = filename
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = None

    def to_tuple(self):
        return (("criteria", tuple(self.criteria)),
                ("nonzero", self.nonzero),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", self.__class__.__name__),
                ("criteria", tuple(self.criteria)),
                ("nonzero", self.nonzero),)

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X))
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def _transform(self, X):
        repertoires = self.get_repertoires(X)
        feature_names = ["present", "total"]
        feature_annotations = pd.DataFrame({"feature": ["present", "total"]})

        encoded = EncodedData(
            examples=repertoires,
            labels=X.encoded_data.labels,
            example_ids=X.encoded_data.example_ids,
            feature_names=feature_names,
            feature_annotations=feature_annotations
        )
        dataset = RepertoireDataset(
            params=X.params,
            encoded_data=encoded,
            filenames=X.get_filenames(),
            identifier=X.identifier
        )
        
        return dataset

    def fit(self, X, y=None):
        return self

    def get_repertoires(self, X):
        if self.nonzero:
            total = X.encoded_data.examples.getnnz(axis=1)
            dataset = DataSummarizer.filter_features(X, self.criteria)
            present = dataset.encoded_data.examples.getnnz(axis=1)
        else:
            total = X.encoded_data.examples.sum(axis=1)
            dataset = DataSummarizer.filter_features(X, self.criteria)
            present = dataset.encoded_data.examples.sum(axis=1)

        repertoires = sparse.csr_matrix(np.column_stack((present, total)))
        return repertoires
