import abc

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReadsType import ReadsType
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedSequencesEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a RepertoireDataset and a reference sequence dataset.

    This encoding can be used in combination with the :ref:`Matches` report.

    When sum_matches and normalize are set to True, this encoder behaves as described in: Yao, Y. et al. ‘T cell receptor repertoire as a potential diagnostic marker for celiac disease’.
    Clinical Immunology Volume 222 (January 2021): 108621. `doi.org/10.1016/j.clim.2020.108621 <https://doi.org/10.1016/j.clim.2020.108621>`_


    Arguments:

        reference (dict): A dictionary describing the reference dataset file. Import should be specified the same way as regular dataset import. It is only allowed to import a sequence dataset here (i.e., is_repertoire and paired are False by default, and are not allowed to be set to True).

        max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the reference sequence.

        reads (:py:mod:`~immuneML.util.ReadsType`): Reads type signify whether the counts of the sequences in the repertoire will be taken into account. If :py:mod:`~immuneML.util.ReadsType.UNIQUE`, only unique sequences (clonotypes) are counted, and if :py:mod:`~immuneML.util.ReadsType.ALL`, the sequence 'count' value is summed when determining the number of matches. The default value for reads is all.

        sum_matches (bool): When sum_matches is False, the resulting encoded data matrix contains multiple columns with the number of matches per reference sequence. When sum_matches is true, all columns are summed together, meaning that there is only one aggregated sum of matches per repertoire in the encoded data.
        To use this encoder in combination with the :ref:`Matches` report, sum_matches must be set to False. When sum_matches is set to True, this encoder behaves as described by Yao, Y. et al. By default, sum_matches is False.

        normalize (bool): If True, the sequence matches are divided by the total number of unique sequences in the repertoire (when reads = unique) or the total number of reads in the repertoire (when reads = all).


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_ms_encoding:
            MatchedSequences:
                reference:
                    path: path/to/file.txt
                    format: VDJDB
                max_edit_distance: 1
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedSequencesRepertoireEncoder"
    }

    def __init__(self, max_edit_distance: int, reference: ReceptorSequenceList, reads: ReadsType, sum_matches: bool, normalize: bool, name: str = None):
        self.max_edit_distance = max_edit_distance
        self.reference_sequences = reference
        self.reads = reads
        self.sum_matches = sum_matches
        self.normalize = normalize
        self.feature_count = 1 if self.sum_matches else len(self.reference_sequences)
        self.name = name

    @staticmethod
    def _prepare_parameters(max_edit_distance: int, reference: dict, reads: str, sum_matches: bool, normalize: bool, name: str = None):
        location = "MatchedSequencesEncoder"

        ParameterValidator.assert_type_and_value(max_edit_distance, int, location, "max_edit_distance", min_inclusive=0)
        ParameterValidator.assert_type_and_value(sum_matches, bool, location, "sum_matches")
        ParameterValidator.assert_type_and_value(normalize, bool, location, "normalize")
        ParameterValidator.assert_in_valid_list(reads.upper(), [item.name for item in ReadsType], location, "reads")

        reference_sequences = MatchedReferenceUtil.prepare_reference(reference_params=reference, location=location, paired=False)

        return {
            "max_edit_distance": max_edit_distance,
            "reference": reference_sequences,
            "reads": ReadsType[reads.upper()],
            "sum_matches": sum_matches,
            "normalize": normalize,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        EncoderHelper.check_dataset_type_available_in_mapping(dataset, MatchedSequencesEncoder)

        prepared_parameters = MatchedSequencesEncoder._prepare_parameters(**params)
        encoder = ReflectionHandler.get_class_by_name(MatchedSequencesEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "reference_encoding/")(**prepared_parameters)

        return encoder

    def encode(self, dataset, params: EncoderParams):

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        encoding_params_desc = {"max_edit_distance": self.max_edit_distance,
                                "reference_sequences": sorted([seq.get_sequence() + seq.metadata.v_gene + seq.metadata.j_gene
                                                               for seq in self.reference_sequences]),
                                "reads": self.reads.name,
                                "sum_matches": self.sum_matches,
                                "normalize": self.normalize}

        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedSequencesEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", encoding_params_desc), )

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass