from sklearn.base import TransformerMixin

from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.analysis.AxisType import AxisType


class CriteriaBasedAnnotation(TransformerMixin):
    """
    Annotates boolean values as either a new label (repertoire annotation) or as a new column in the
    feature_annotations based matching of the criteria as defined in CriteriaMatcher class.
    This is a step in the PipelineEncoder.
    """

    def __init__(self, axis: AxisType, criteria, annotation_name="criteria_annotation", result_path=None, filename=None):
        self.axis = axis
        self.criteria = criteria
        self.annotation_name = annotation_name
        self.result_path = result_path
        self.filename = filename

    def transform(self, X):

        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.annotate_repertoires(X, self.criteria, self.annotation_name)
        else:
            dataset = DataSummarizer.annotate_features(X, self.criteria, self.annotation_name)

        return dataset

    def fit(self, X, y=None):
        return self
