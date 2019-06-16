from sklearn.base import TransformerMixin

from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.analysis.AxisType import AxisType


class CriteriaBasedFilter(TransformerMixin):
    """
    Filters either repertoires or features based matching of the criteria as defined in CriteriaMatcher class.
    This is a step in the PipelineEncoder.
    """
    def __init__(self, axis: AxisType, criteria, result_path=None, filename=None):
        self.axis = axis
        self.criteria = criteria
        self.result_path = result_path
        self.filename = filename
        self.selected_indices = None

    def transform(self, X):

        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.filter_repertoires(X, self.criteria)
        else:
            dataset = DataSummarizer.filter_features(X, self.criteria)

        return dataset

    def fit(self, X, y=None):
        return self
