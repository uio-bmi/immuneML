from sklearn.base import TransformerMixin

from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.analysis.data_manipulation.GroupSummarizationType import GroupSummarizationType
from source.analysis.AxisType import AxisType

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

    def transform(self, X):

        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.group_repertoires(X, self.group_columns, self.group_summarization_type)
        else:
            dataset = DataSummarizer.group_features(X, self.group_columns, self.group_summarization_type)

        return dataset

    def fit(self, X, y=None):
        return self
