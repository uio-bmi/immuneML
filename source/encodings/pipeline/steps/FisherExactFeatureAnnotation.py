import copy
import math
import os
from multiprocessing.pool import Pool

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.analysis.criteria_matches.CriteriaMatcher import CriteriaMatcher
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.pipeline.steps.FisherExactWrapper import FisherExactWrapper


class FisherExactFeatureAnnotation(TransformerMixin):
    """
    Annotates results of a Fisher's exact test for each unique feature in the dataset by adding new columns to the
    feature_annotations dataset. To make it binary, you must specify the positive_criteria, as defined in the CriteriaMatcher class.
    This is a step in the PipelineEncoder.
    """
    def __init__(self, positive_criteria, result_path=None, filename=None):
        self.positive_criteria = positive_criteria
        self.result_path = result_path
        self.filename = filename
        self.fisher_annotations = None
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = ""

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def to_tuple(self):
        return (("axis", tuple(self.positive_criteria)),
                ("fisher_annotations", tuple(self.filename)),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", FisherExactFeatureAnnotation.__name__),
                ("positive_criteria", str(self.positive_criteria)),)

    def _transform(self, X):
        if not any(["fisher" in column for column in X.encoded_data.feature_annotations.columns]):
            feature_annotations = pd.merge(X.encoded_data.feature_annotations,
                                           self.fisher_annotations,
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
            dataset.encoded_data.feature_annotations.to_csv(self.result_path + "/feature_annotations.csv")
            FisherExactFeatureAnnotation.store(dataset, self.result_path, self.filename)
        else:
            dataset = copy.deepcopy(X)
        return dataset

    def fit(self, X: Dataset, y=None):
        if not any(["fisher" in column for column in X.encoded_data.feature_annotations.columns]):
            repertoire_classes = FisherExactFeatureAnnotation.get_positive(X, self.positive_criteria)
            self.fisher_annotations = FisherExactFeatureAnnotation.compute_fisher_annotations(X, repertoire_classes)
        return self

    @staticmethod
    def get_positive(X: Dataset, positive_criteria):
        data = pd.DataFrame(X.encoded_data.labels)
        matcher = CriteriaMatcher()
        results = matcher.match(criteria=positive_criteria, data=data)
        return results

    @staticmethod
    def compute_fisher_annotations(X: Dataset, repertoire_classes):
        feature_chunks = FisherExactFeatureAnnotation.create_chunks(X.encoded_data.repertoires, X.encoded_data.feature_names)
        args = [(feature_chunk, repertoire_classes, FisherExactWrapper()) for feature_chunk in feature_chunks]
        with Pool(os.cpu_count()) as pool:
            results = pool.starmap(FisherExactFeatureAnnotation.compute_fisher, args)
        results = [item for sublist in results for item in sublist]
        fisher_annotations = pd.DataFrame(results)
        keep_same = ['feature']
        fisher_annotations.columns = ['{}{}'.format('' if c in keep_same else 'fisher_', c) for c in
                                      fisher_annotations.columns]
        return fisher_annotations

    @staticmethod
    def create_chunks(repertoires, features):
        n_col = repertoires.shape[1]
        n = math.ceil(n_col / os.cpu_count())
        chunks = [
            (range(i, i + n if i + n < n_col else n_col),
             [features[x] for x in range(i, i + n if i + n < n_col else n_col)],
             repertoires[:, i:(i + n if i + n < n_col else n_col)]) for i in range(0, n_col, n)
        ]
        return chunks

    @staticmethod
    def compute_fisher(sequence_chunk, sample_labels, fisher_exact: FisherExactWrapper):
        indices, features, matrix = sequence_chunk
        results = []
        for original_index, (new_index, feature) in zip(indices, enumerate(features)):
            presence_absence = matrix[:, new_index].A[:, 0] > 0
            conf_mat = confusion_matrix(presence_absence, sample_labels, labels=[False, True])
            p = fisher_exact.fisher_exact(conf_mat, FisherExactWrapper.TWO_SIDED)
            results.append({
                "feature": feature, 
                "index": original_index, 
                "p.two_tail": p,
                "pres_0": conf_mat[1, 0],
                "pres_1": conf_mat[1, 1],
                "abs_0": conf_mat[0, 0],
                "abs_1": conf_mat[0, 1]
            })
        return results

    @staticmethod
    def store(encoded_dataset: Dataset, result_path, filename):
        PickleExporter.export(encoded_dataset, result_path, filename)
