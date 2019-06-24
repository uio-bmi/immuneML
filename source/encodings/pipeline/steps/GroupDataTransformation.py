from sklearn.base import TransformerMixin

from source.analysis.AxisType import AxisType
from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.analysis.data_manipulation.GroupSummarizationType import GroupSummarizationType
from source.caching.CacheHandler import CacheHandler


class GroupDataTransformation(TransformerMixin):
    """
    Groups together encoded data values by either grouping multiple repertoires together or multiple features together.
    One example of grouping repertoires together is averaging all repertoires with the same disease status to get average
    per group. One example of grouping features together is grouping all matches to a single reference sequence after
    annotating your dataset with Ag-specific sequences from a reference database.
    This is a step in the PipelineEncoder.
    """
    def __init__(self, axis: AxisType, group_columns, group_summarization_type: GroupSummarizationType, result_path=None, filename=None):
        self.axis = axis
        self.group_columns = group_columns
        self.group_summarization_type = group_summarization_type
        self.result_path = result_path
        self.filename = filename
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = ""

    def to_tuple(self):
        return (("axis", self.axis),
                ("group_columns", tuple(self.group_columns)),
                ("group_summarization_type", self.group_summarization_type),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", GroupDataTransformation.__name__),
                ("axis", self.axis),
                ("group_columns", self.group_columns),
                ("group_summarization_type", self.group_summarization_type),)

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def _transform(self, X):

        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.group_repertoires(X, self.group_columns, self.group_summarization_type)
        else:
            dataset = DataSummarizer.group_features(X, self.group_columns, self.group_summarization_type)

        return dataset

    def fit(self, X, y=None):
        return self
