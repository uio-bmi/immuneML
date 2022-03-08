import abc
from typing import List

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.receptor.BCReceptor import BCReceptor
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.TCGDReceptor import TCGDReceptor
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReadsType import ReadsType
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedReceptorsEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a dataset containing unpaired (single chain) data,
    and a paired reference receptor dataset.
    For each paired reference receptor, the frequency of either chain in the dataset is counted.

    This encoding can be used in combination with the :ref:`Matches` report.

    When sum_matches and normalize are set to True, this encoder behaves similarly as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
    Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_
    with the only exception being that this encoder uses paired receptors, while the original publication used single sequences (see also: :ref:`MatchedSequences` encoder).

    Arguments:

        reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and paired is True by default, and these are not allowed to be changed).

        max_edit_distances (dict): A dictionary specifying the maximum edit distance between a target sequence (from the repertoire) and the reference sequence. A maximum distance can be specified per chain, for example to allow for less strict matching of TCR alpha and BCR light chains. When only an integer is specified, this distance is applied to all possible chains.

        reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when determining the number of matches. The default value for reads is all.

        sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with the number of matches per reference receptor chain. When sum_matches is true, the columns representing each of the two chains are summed together, meaning that there are only two aggregated sums of matches (one per chain) per repertoire in the encoded data.
        To use this encoder in combination with the :ref:`Matches` report, sum_matches must be set to False. When sum_matches is set to True, this encoder behaves similarly to the encoder described by Yao, Y. et al. By default, sum_matches is False.

        normalize (bool): If True, the chain matches are divided by the total number of unique receptors in the repertoire (when reads = unique) or the total number of reads in the repertoire (when reads = all).


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_mr_encoding:
            MatchedReceptors:
                reference:
                    path: path/to/file.txt
                    format: VDJDB
                max_edit_distances:
                    alpha: 1
                    beta: 0
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedReceptorsRepertoireEncoder"
    }

    def __init__(self, reference: List[Receptor], max_edit_distances: dict, reads: ReadsType, sum_matches: bool, normalize: bool, name: str = None):
        self.reference_receptors = reference
        self.max_edit_distances = max_edit_distances
        self.reads = reads
        self.sum_matches = sum_matches
        self.normalize = normalize
        self.feature_count = 2 if self.sum_matches else len(self.reference_receptors) * 2
        self.name = name

    @staticmethod
    def _prepare_parameters(reference: dict, max_edit_distances: dict, reads: str, sum_matches: bool, normalize: bool, name: str = None):
        location = "MatchedReceptorsEncoder"

        ParameterValidator.assert_type_and_value(sum_matches, bool, location, "sum_matches")
        ParameterValidator.assert_type_and_value(normalize, bool, location, "normalize")
        ParameterValidator.assert_in_valid_list(reads.upper(), [item.name for item in ReadsType], location, "reads")

        legal_chains = [chain for receptor in (TCABReceptor(), TCGDReceptor(), BCReceptor()) for chain in receptor.get_chains()]

        if type(max_edit_distances) is int:
            max_edit_distances = {chain: max_edit_distances for chain in legal_chains}
        elif type(max_edit_distances) is dict:
            ParameterValidator.assert_keys(max_edit_distances.keys(), legal_chains, location, "max_edit_distances", exclusive=False)
        else:
            ParameterValidator.assert_type_and_value(max_edit_distances, dict, location, 'max_edit_distances')

        reference_receptors = MatchedReferenceUtil.prepare_reference(reference, location=location, paired=True)

        return {
            "reference": reference_receptors,
            "max_edit_distances": max_edit_distances,
            "reads": ReadsType[reads.upper()],
            "sum_matches": sum_matches,
            "normalize": normalize,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        EncoderHelper.check_dataset_type_available_in_mapping(dataset, MatchedReceptorsEncoder)

        prepared_params = MatchedReceptorsEncoder._prepare_parameters(**params)
        encoder = ReflectionHandler.get_class_by_name(MatchedReceptorsEncoder.dataset_mapping[dataset.__class__.__name__], "reference_encoding/")(**prepared_params)

        return encoder

    def encode(self, dataset, params: EncoderParams):
        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        chains = [(receptor.get_chain(receptor.get_chains()[0]), receptor.get_chain(receptor.get_chains()[1]))
                  for receptor in self.reference_receptors]

        encoding_params_desc = {"max_edit_distance": sorted(self.max_edit_distances.items()),
                                "reference_receptors": sorted([chain_a.get_sequence() + chain_a.metadata.v_gene + chain_a.metadata.j_gene + "|" +
                                                                chain_b.get_sequence() + chain_b.metadata.v_gene + chain_b.metadata.j_gene
                                                                for chain_a, chain_b in chains]),
                                "reads": self.reads.name,
                                "sum_matches": self.sum_matches,
                                "normalize": self.normalize}

        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedReceptorsEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", encoding_params_desc), )

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass