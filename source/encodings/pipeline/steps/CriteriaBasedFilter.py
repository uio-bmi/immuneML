from sklearn.base import TransformerMixin

from source.analysis.AxisType import AxisType
from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.caching.CacheHandler import CacheHandler


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
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = ""

    def to_tuple(self):
        return (("axis", self.axis),
                ("criteria", tuple(self.criteria)),
                ("selected_indices", tuple(self.selected_indices)),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", CriteriaBasedFilter.__name__),
                ("axis", self.axis),
                ("criteria", self.criteria),
                ("selected_indices", self.selected_indices),)

    def _transform(self, X):

        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.filter_repertoires(X, self.criteria)
        else:
            dataset = DataSummarizer.filter_features(X, self.criteria)

        return dataset

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def fit(self, X, y=None):
        return self
