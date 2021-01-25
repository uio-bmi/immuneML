import abc
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


class MatchedRegexEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a RepertoireDataset and a collection of regular expressions.
    For each regular expression, the number of sequences in the RepertoireDataset containing the expression is counted.
    This can also be used to count how often a subsequence occurs in a RepertoireDataset.

    The regular expressions are defined per chain, and it is possible to require a V gene match in addition to the
    CDR3 sequence containing the regular expression.

    This encoding should be used in combination with the :ref:`Matches`
    report.


    Arguments:

        match_v_genes (bool): Whether V gene matches are required. If this is True, a match is only counted if the
        V gene matches the gene specified in the motif input file. By default match_v_genes is False.

        sum_counts (bool): When counting the number of matches, one can choose to count the number of matching sequences
        or sum the frequencies of those sequences. If sum_counts is True, the sequence frequencies are summed. Otherwise,
        if sum_counts is False, the number of matching unique sequences is counted. By default sum_counts is False.

        motif_filepath (str): The path to the motif input file. This should be a tab separated file containing a
        column named 'id' and for every chain that should be matched a column containing the regex (<chain>_regex) and a column containing
        the V gene (<chain>V) if match_v_genes is True.
        The chains are specified by their three letter code, see :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`.

        In the simplest case, when counting the number of occurrences of a given list of k-mers in TRB sequences, the contents of the motif file could look like this:

            ====  ==========
            id    TRB_regex
            ====  ==========
            1     ACG
            2     EDNA
            3     DFWG
            ====  ==========

        It is also possible to test whether paired regular expressions occur in the dataset (for example: regular expressions
        matching both a TRA chain and a TRB chain) by specifying them on the same line.
        In a more complex case where both paired and unpaired regular expressions are specified, in addition to matching the V
        genes, the contents of the motif file could look like this:

            ====  ==========  =======  ==========  ========
            id    TRA_regex   TRAV     TRB_regex   TRBV
            ====  ==========  =======  ==========  ========
            1     AGQ.GSS     TRAV35   S[APL]GQY   TRBV29-1
            2                          ASS.R.*     TRBV7-3
            ====  ==========  =======  ==========  ========


    YAML Specification:

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

    def __init__(self, motif_filepath: Path, match_v_genes: bool, sum_counts: bool, chains: list, name: str = None):
        self.motif_filepath = motif_filepath
        self.match_v_genes = match_v_genes
        self.sum_counts = sum_counts
        self.chains = chains
        self.regex_df = None
        self.feature_count = None
        self.name = name

    @staticmethod
    def _prepare_parameters(motif_filepath: str, match_v_genes: bool, sum_counts: bool, name: str = None):

        ParameterValidator.assert_type_and_value(match_v_genes, bool, "MatchedRegexEncoder", "match_v_genes")
        ParameterValidator.assert_type_and_value(sum_counts, bool, "MatchedRegexEncoder", "sum_counts")

        motif_filepath = Path(motif_filepath)
        assert motif_filepath.is_file(), f"MatchedRegexEncoder: the file {motif_filepath} does not exist. " \
                                               f"Specify the correct path under motif_filepath."

        file_columns = list(pd.read_csv(motif_filepath, sep="\t", iterator=False, dtype=str, nrows=0).columns)

        ParameterValidator.assert_all_in_valid_list(file_columns, ["id"] + [f"{c.value}V" for c in Chain] + [f"{c.value}_regex" for c in Chain], "MatchedRegexEncoder", "motif_filepath (column names)")

        chains = [colname.split("_")[0] for colname in file_columns if colname.endswith("_regex")]
        if match_v_genes:
            for chain in chains:
                assert f"{chain}V" in file_columns, f"MatchedRegexEncoder: expected column {chain}V to be present in the columns of motif_filepath. " \
                                                    f"Remove {chain}_regex from columns, or set match_v_genes to False."

        return {
            "motif_filepath": motif_filepath,
            "match_v_genes": match_v_genes,
            "sum_counts": sum_counts,
            "chains": chains,
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
        df = pd.read_csv(self.motif_filepath, sep="\t", iterator=False, dtype=str)

        if not self.match_v_genes:
            for v_gene in [f"{c.value}V" for c in Chain]:
                if v_gene in df.columns:
                    df.drop(v_gene, axis=1, inplace=True)

        colnames_subset = list(df.columns)
        colnames_subset.remove("id")
        df.drop_duplicates(subset=colnames_subset, inplace=True)

        df.replace({np.NaN: None}, inplace=True)

        self.feature_count = 0

        for chain in Chain:
            regex_colname = f"{chain.value}_regex"
            if regex_colname in df.columns:
                self.feature_count += df[regex_colname].count()

        self.regex_df = df


    @staticmethod
    def get_documentation():
        doc = str(MatchedRegexEncoder.__doc__)

        chain_values = str([region_type.value for region_type in Chain])[1:-1].replace("'", "`")

        mapping = {
            "The chains are specified by their three letter code, see :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`.": f"The chains are specified by their three letter code, valid values are: {chain_values}.",
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc



