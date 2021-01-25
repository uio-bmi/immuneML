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
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedReceptorsEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a dataset containing unpaired (single chain) data,
    and a paired reference receptor dataset.
    For each paired reference receptor, the frequency of either chain in the dataset is counted.

    This encoding should be used in combination with the :ref:`Matches`
    report.

    Arguments:

        reference (dict): A dictionary describing the reference dataset file, specified the same as regular data import.
        See the :py:mod:`immuneML.IO.sequence_import` for specification details.
        Must contain paired receptor sequences.

        max_edit_distances (dict): A dictionary specifying the maximum edit distance between a target sequence
        (from the repertoire) and the reference sequence. A maximum distance can be specified per chain, for example
        to allow for less strict matching of TCR alpha and BCR light chains. When only an integer is specified,
        this distance is applied to all possible chains.


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_mr_encoding:
            MatchedReceptors:
                reference:
                    format: IRIS
                    params:
                        path: /path/to/file.txt
                        paired: True
                        all_dual_chains: True
                        all_genes: True
                max_edit_distances:
                    alpha: 1
                    beta: 0
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedReceptorsRepertoireEncoder"
    }

    def __init__(self, reference_receptors: List[Receptor], max_edit_distances: dict, name: str = None):
        self.reference_receptors = reference_receptors
        self.max_edit_distances = max_edit_distances
        self.name = name

    @staticmethod
    def _prepare_parameters(reference: dict, max_edit_distances: dict, name: str = None):
        location = "MatchedReceptorsEncoder"

        legal_chains = [chain for receptor in (TCABReceptor(), TCGDReceptor(), BCReceptor()) for chain in receptor.get_chains()]

        if type(max_edit_distances) is int:
            max_edit_distances = {chain: max_edit_distances for chain in legal_chains}
        elif type(max_edit_distances) is dict:
            ParameterValidator.assert_keys(max_edit_distances.keys(), legal_chains, location, "max_edit_distances", exclusive=False)
        else:
            ParameterValidator.assert_type_and_value(max_edit_distances, dict, location, 'max_edit_distances')

        reference_receptors = MatchedReferenceUtil.prepare_reference(reference, location=location, paired=True)

        return {
            "reference_receptors": reference_receptors,
            "max_edit_distances": max_edit_distances,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_params = MatchedReceptorsEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(
                MatchedReceptorsEncoder.dataset_mapping[dataset.__class__.__name__], "reference_encoding/")(**prepared_params)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(MatchedReceptorsEncoder.__name__,
                                                                                dataset.__class__.__name__))
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
                                                                for chain_a, chain_b in chains])}

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