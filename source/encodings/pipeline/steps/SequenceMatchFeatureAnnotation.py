import os
from multiprocessing.pool import Pool

import dask.dataframe as dd
import numpy as np
import pandas as pd
import regex as re
from dask.multiprocessing import get
from editdistance import eval
from sklearn.base import TransformerMixin

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.util.ReflectionHandler import ReflectionHandler


class SequenceMatchFeatureAnnotation(TransformerMixin):
    """
    Annotates features based on a reference dataset of sequences, with experimental data, such as antigen specificity,
    associated pathologies, etc.
    This is a step in the PipelineEncoder.

    Use cases:
     query sequence matches reference sequence
     TODO: query sequence contains reference motif
     TODO: query motif is within reference sequence
     TODO: query motif matches reference motif
    """

    FEATURE = "feature"
    SEQUENCE = "sequence"

    def __init__(self, reference_sequence_path, data_loader_params, sequence_matcher_params,
                 data_loader_name="GenericLoader", annotation_prefix="", result_path=None, filename=None):
        """
        :param reference_sequence_path: path to sequence dataset containing experimental sequence-level data
        :param data_loader_params: params for GenericLoader to be used for loading the sequence data
        :param sequence_matcher_params: dict with keys for:
                - metadata_fields_to_match which specify the values (such as v_gene, etc), which must be identical to be
                considered a match;
                - max_distance: the max LD in the sequence to be considered a match;
                - same_length: boolean specifying whether the match must have a sequence of same length
        :param data_loader_name: class name
        :param annotation_prefix:
        :param result_path: where to store the result
        :param filename: filename
        """
        mod = ReflectionHandler.get_class_by_name(data_loader_name, "IO")
        self.data_loader = mod()
        self.reference_sequence_path = reference_sequence_path
        self.data_loader_params = data_loader_params
        self.sequence_matcher_params = sequence_matcher_params
        self.annotation_prefix = annotation_prefix
        self.result_path = result_path
        self.filename = filename
        self.initial_encoder = ""
        self.initial_encoder_params = ""
        self.previous_steps = None

    def fit(self, X, y=None):
        return self

    def to_tuple(self):
        return (("data_loader", self.data_loader.__class__.__name__),
                ("reference_sequence_path", self.reference_sequence_path),
                ("sequence_matcher_params", tuple([(key, self.sequence_matcher_params[key])
                                                   for key in self.sequence_matcher_params.keys()])),
                ("annotation_prefix", self.annotation_prefix),
                ("encoding_step", self.__class__.__name__),)

    def _prepare_caching_params(self, dataset):
        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("encoding", "PipelineEncoder"),
                ("initial_encoder", self.initial_encoder),
                ("initial_encoder_params", self.initial_encoder_params),
                ("previous_steps", self.previous_steps),
                ("encoding_step", self.__class__.__name__),
                ("data_loader", self.data_loader.__class__.__name__),
                ("reference_sequence_path", self.reference_sequence_path),
                ("sequence_matcher_params", tuple([(key, self.sequence_matcher_params[key])
                                                   for key in self.sequence_matcher_params.keys()])),
                ("annotation_prefix", self.annotation_prefix),)

    def transform(self, X):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(X), "")
        dataset = CacheHandler.memo(cache_key, lambda: self._transform(X))
        return dataset

    def annotate(self, X: Dataset):
        match_annotations = self.compute_match_annotations(X)
        match_annotations.to_csv(self.result_path + "match_annotations.csv")
        feature_annotations = pd.merge(X.encoded_data.feature_annotations,
                                       match_annotations,
                                       on=[self.FEATURE] + self.sequence_matcher_params["metadata_fields_to_match"] + [
                                           self.SEQUENCE],
                                       how='left')
        encoded = EncodedData(
            repertoires=X.encoded_data.repertoires,
            labels=X.encoded_data.labels,
            repertoire_ids=X.encoded_data.repertoire_ids,
            feature_names=X.encoded_data.feature_names,
            feature_annotations=feature_annotations
        )
        return Dataset(
            params=X.params,
            encoded_data=encoded,
            filenames=X.get_filenames(),
            identifier=X.id,
            metadata_file=X.metadata_file
        )

    def is_annotated(self, X):
        return (self.annotation_prefix is not "" and any(
            [self.annotation_prefix in column for column in X.encoded_data.feature_annotations.columns]))

    def _transform(self, X):
        if not self.is_annotated(X):
            dataset = self.annotate(X)
        else:
            dataset = X
        dataset.encoded_data.feature_annotations.to_csv(self.result_path + "/feature_annotations.csv")
        self.store(dataset, self.result_path, self.filename)
        return dataset

    def compute_match_annotations(self, X):
        reference_sequences = self.prepare_reference_sequences()
        dataset_sequences = X.encoded_data.feature_annotations
        sequence_matches= self.match(dataset_sequences, reference_sequences, **self.sequence_matcher_params)
        merge_on = self.sequence_matcher_params["metadata_fields_to_match"] + [self.SEQUENCE]
        match_annotations = pd.merge(dataset_sequences, sequence_matches, how="left", on=merge_on)
        match_annotations.columns = match_annotations.columns.str.replace("^matched_",
                                                                          self.annotation_prefix + "matched_")
        match_annotations = match_annotations.fillna(value=np.nan)
        match_annotations = match_annotations.replace('None', np.nan)
        return match_annotations

    def prepare_reference_sequences(self):
        dataset = self.data_loader.load(path=self.reference_sequence_path, params=self.data_loader_params)
        reference_sequences = []
        for repertoire in dataset.get_data():
            reference_sequences.extend(repertoire.sequences)
        reference_sequences_df = []
        for sequence in reference_sequences:
            metadata = vars(sequence.metadata)
            metadata = {key: value for key, value in metadata.items()
                        if key in self.sequence_matcher_params["metadata_fields_to_match"]}
            reference_sequences_df.append({**metadata, **sequence.metadata.custom_params,
                                           self.SEQUENCE: sequence.get_sequence()})
        return pd.DataFrame(reference_sequences_df)

    def match_regex(self, rx, value):
        return bool(rx.match(value))

    def store(self, encoded_dataset: Dataset, result_path, filename):
        PickleExporter.export(encoded_dataset, result_path, filename)

    def filter_query(self, query, reference, max_distance, same_length):
        error_type = "s" if same_length else "e"
        reference_sequences = "|".join(reference["sequence"])
        regex_str = "({reference_sequence}){{{error_type}<={max_distance}}}"
        regex_str = regex_str.format(reference_sequence=reference_sequences, error_type=error_type,
                                     max_distance=str(max_distance))
        rx = re.compile(regex_str)
        arguments = [(rx, sequence) for sequence in query["sequence"].values]
        with Pool(os.cpu_count()) as pool:
            any_match = pool.starmap(self.match_regex, arguments)
        return query[any_match]

    def match_sequence(self, x, references, metadata_fields_to_match, max_distance, same_length):
        indices = np.ones(references.shape[0])
        if same_length:
            indices = np.logical_and(indices, (references[self.SEQUENCE].str.len() == len(x[self.SEQUENCE])).values)
        if len(metadata_fields_to_match) > 0:
            for column in metadata_fields_to_match:
                indices = np.logical_and(indices, (references[column] == x[column]).values)
        matches = [sequence for sequence in references.loc[indices, self.SEQUENCE]
                   if eval(x[self.SEQUENCE], sequence) <= max_distance]
        if len(matches) > 0:
            return matches[0]

    def match_dataset(self, query, reference, metadata_fields_to_match, max_distance, same_length):
        query_dd = dd.from_pandas(query, npartitions=os.cpu_count())
        query['matched_sequence'] = query_dd.map_partitions(lambda df:
                                                            df.apply((lambda row: self.match_sequence(row, reference,
                                                             metadata_fields_to_match, max_distance, same_length)),
                                                                     axis=1)).compute(scheduler=get)
        reference.columns = ["matched_" + column for column in reference.columns]
        result = pd.merge(query, reference, on="matched_sequence", how="left")
        return result

    def match(self, query, reference, metadata_fields_to_match, max_distance, same_length):
        columns = [self.SEQUENCE] + metadata_fields_to_match
        query = query[columns]
        query = self.filter_query(query, reference, max_distance, same_length)
        return self.match_dataset(query, reference, metadata_fields_to_match, max_distance, same_length)
