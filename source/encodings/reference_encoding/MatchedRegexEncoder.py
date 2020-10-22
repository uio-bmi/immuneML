import abc
import os
import pandas as pd
import numpy as np
from source.caching.CacheHandler import CacheHandler
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MatchedRegexEncoder(DatasetEncoder):
    """
    todo fill in
    Arguments:
        motif_filepath (str): todo fill in
        match_v_genes (bool): todo fill in
    Specification:
    .. indent with spaces
    .. code-block:: yaml
        my_mr_encoding:
            MatchedRegex:
                motif_filepath: /path/to/file.txt
                match_v_genes: True
                sum_counts: False
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedRegexRepertoireEncoder"
    }

    def __init__(self, motif_filepath: str, match_v_genes: bool, sum_counts: bool, name: str = None):
        self.motif_filepath = motif_filepath
        self.match_v_genes = match_v_genes
        self.sum_counts = sum_counts
        self.regex_df = None
        self.feature_count = None
        self.name = name

    @staticmethod
    def _prepare_parameters(motif_filepath: str, match_v_genes: bool, sum_counts: bool, name: str = None):

        ParameterValidator.assert_type_and_value(match_v_genes, bool, "MatchedRegexEncoder", "match_v_genes")
        ParameterValidator.assert_type_and_value(sum_counts, bool, "MatchedRegexEncoder", "sum_counts")

        assert os.path.isfile(motif_filepath), f"MatchedRegexEncoder: the file {motif_filepath} does not exist. " \
                                               f"Specify the correct path under motif_filepath."

        # todo test that correct headers in df

        return {
            "motif_filepath": motif_filepath,
            "match_v_genes": match_v_genes,
            "sum_counts": sum_counts,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_params = MatchedRegexEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(
                MatchedRegexEncoder.dataset_mapping[dataset.__class__.__name__], "reference_encoding/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(MatchedRegexEncoder.__name__,
                                                                                dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):
        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedRegexEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", tuple(vars(self).items())))

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    def _load_regex_df(self):
        # todo make column mapping more general later (work with BCR as well, etc)
        df = pd.read_csv(self.motif_filepath, sep="\t", iterator=False, dtype=str)

        if not self.match_v_genes:
            df.drop(["TRAV", "TRBV"], axis=1, inplace=True)

        colnames_subset = list(df.columns)
        colnames_subset.remove("id")
        df.drop_duplicates(subset=colnames_subset, inplace=True)

        df.replace({np.NaN: None}, inplace=True)

        self.feature_count = df["TRA_regex"].count() + df["TRB_regex"].count()

        self.regex_df = df