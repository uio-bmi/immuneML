import pandas as pd
from sklearn.base import TransformerMixin

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData


class PublicSequenceFeatureAnnotation(TransformerMixin):
    """
    Annotates, for each feature, the number of repertoires that the feature is present in. This is added as a column
    to encoded_dataset.feature_annotations. 
    This is a step in the PipelineEncoder.
    """

    COLUMNS_PUBLIC = "public"
    PUBLIC_REPERTOIRE_COUNT = "public_number_of_repertoires"
    FEATURE = "feature"

    def __init__(self, result_path=None, filename=None):
        self.result_path = result_path
        self.filename = filename
        self.public_annotations = None
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = None

    def to_tuple(self):
        return ("encoding_step", self.__class__.__name__)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", self.__class__.__name__),)

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def _annotate_public_features(self, X: Dataset):
        feature_annotations = pd.merge(X.encoded_data.feature_annotations,
                                       self.public_annotations,
                                       on="feature",
                                       how='left')
        encoded = EncodedData(
            repertoires=X.encoded_data.repertoires,
            labels=X.encoded_data.labels,
            repertoire_ids=X.encoded_data.repertoire_ids,
            feature_names=X.encoded_data.feature_names,
            feature_annotations=feature_annotations
        )
        dataset = Dataset(
            data=X.data,
            params=X.params,
            encoded_data=encoded,
            filenames=X.get_filenames(),
            identifier=X.id,
            metadata_file=X.metadata_file
        )
        return dataset

    def _transform(self, X):
        if not any([self.COLUMNS_PUBLIC in column for column in X.encoded_data.feature_annotations.columns]):
            dataset = self._annotate_public_features(X)
            dataset.encoded_data.feature_annotations.to_csv(self.result_path + "/feature_annotations.csv")
            self.store(dataset, self.result_path, self.filename)
            return dataset
        else:
            return X

    def fit(self, X, y=None):
        if not any([self.COLUMNS_PUBLIC in column for column in X.encoded_data.feature_annotations.columns]):
            self.public_annotations = self.compute_public_annotations(X)
        return self

    def compute_public_annotations(self, X):
        sums = X.encoded_data.repertoires.getnnz(axis=0)
        public_annotations = pd.DataFrame({PublicSequenceFeatureAnnotation.FEATURE: X.encoded_data.feature_names,
                                           PublicSequenceFeatureAnnotation.PUBLIC_REPERTOIRE_COUNT: sums})
        return public_annotations

    def store(self, encoded_dataset: Dataset, result_path, filename):
        PickleExporter.export(encoded_dataset, result_path, filename)
