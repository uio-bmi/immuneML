from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from scipy import sparse

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.encodings.EncoderParams import EncoderParams


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

    def transform(self, X):
        repertoires = self.get_repertoires(X)
        feature_names = ["present", "total"]
        feature_annotations = pd.DataFrame({"feature": ["present", "total"]})

        encoded = EncodedData(
            repertoires=repertoires,
            labels=X.encoded_data.labels,
            repertoire_ids=X.encoded_data.repertoire_ids,
            feature_names=feature_names,
            feature_annotations=feature_annotations
        )
        dataset = Dataset(
            data=X.data,
            params=X.params,
            encoded_data=encoded,
            filenames=X.get_filenames(),
            identifier=X.id
        )
        
        return dataset

    def fit(self, X, y=None):
        return self

    def get_repertoires(self, X):
        if self.nonzero:
            total = X.encoded_data.repertoires.getnnz(axis=1)
            dataset = DataSummarizer.filter_features(X, self.criteria)
            present = dataset.encoded_data.repertoires.getnnz(axis=1)
        else:
            total = X.encoded_data.repertoires.sum(axis=1)
            dataset = DataSummarizer.filter_features(X, self.criteria)
            present = dataset.encoded_data.repertoires.sum(axis=1)

        repertoires = sparse.csr_matrix(np.column_stack((present, total)))
        return repertoires
