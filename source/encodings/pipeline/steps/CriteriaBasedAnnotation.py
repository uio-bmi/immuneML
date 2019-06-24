from sklearn.base import TransformerMixin

from source.analysis.AxisType import AxisType
from source.analysis.data_manipulation.DataSummarizer import DataSummarizer
from source.caching.CacheHandler import CacheHandler


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
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = ""

    def to_tuple(self):
        return (("axis", self.axis),
                ("criteria", tuple(self.criteria)),
                ("annotation_name", self.annotation_name),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", CriteriaBasedAnnotation.__name__),
                ("axis", self.axis),
                ("criteria", self.criteria),
                ("annotation_name", self.annotation_name),)

    def _transform(self, X):
        if self.axis == AxisType.REPERTOIRES:
            dataset = DataSummarizer.annotate_repertoires(X, self.criteria, self.annotation_name)
        else:
            dataset = DataSummarizer.annotate_features(X, self.criteria, self.annotation_name)
        return dataset

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def fit(self, X, y=None):
        return self
