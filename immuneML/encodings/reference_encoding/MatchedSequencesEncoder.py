import abc

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReferenceUtil import MatchedReferenceUtil
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MatchedSequencesEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a RepertoireDataset and a reference sequence dataset.

    This encoding should be used in combination with the :ref:`Matches` report.


    Arguments:

        reference (dict): A dictionary describing the reference dataset file. See the :py:mod:`immuneML.IO.sequence_import` for specification details.

        max_edit_distance (dict): The maximum edit distance between a target sequence (from the repertoire) and the reference sequence. A maximum distance can be specified per chain.


    YAML Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_ms_encoding:
            MatchedSequences:
                reference:
                    path: /path/to/file.txt
                    format: VDJDB
                max_edit_distance: 1
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedSequencesRepertoireEncoder"
    }

    def __init__(self, max_edit_distance: int, reference_sequences: ReceptorSequenceList, name: str = None):
        self.max_edit_distance = max_edit_distance
        self.reference_sequences = reference_sequences
        self.name = name

    @staticmethod
    def _prepare_parameters(max_edit_distance: int, reference: dict, name: str = None):
        location = "MatchedSequencesEncoder"

        ParameterValidator.assert_type_and_value(max_edit_distance, int, location, "max_edit_distance", min_inclusive=0)

        reference_sequences = MatchedReferenceUtil.prepare_reference(reference_params=reference, location=location, paired=False)

        return {
            "max_edit_distance": max_edit_distance,
            "reference_sequences": reference_sequences,
            "name": name
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_parameters = MatchedSequencesEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(MatchedSequencesEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "reference_encoding/")(**prepared_parameters)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(MatchedSequencesEncoder.__name__,
                                                                                dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        encoding_params_desc = {"max_edit_distance": self.max_edit_distance,
                                "reference_sequences": sorted([seq.get_sequence() + seq.metadata.v_gene + seq.metadata.j_gene
                                                               for seq in self.reference_sequences])}

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