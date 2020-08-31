import abc
import os

from source.caching.CacheHandler import CacheHandler
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceEncoder(DatasetEncoder):
    """
    Legacy? To be removed?
    """

    dataset_mapping = {
        "RepertoireDataset": "ReferenceRepertoireEncoder"
    }

    def __init__(self, max_edit_distance: int, summary: SequenceMatchingSummaryType, reference_sequences: ReceptorSequenceList, name: str = None):
        self.max_edit_distance = max_edit_distance
        self.summary = summary
        self.reference_sequences = reference_sequences
        self.name = name

    @staticmethod
    def _prepare_parameters(max_edit_distance: int, summary: str, reference_sequences: dict):
        location = "MatchedReferenceEncoder"

        ParameterValidator.assert_type_and_value(max_edit_distance, int, location, "max_edit_distance", min_inclusive=0)
        ParameterValidator.assert_keys(list(reference_sequences.keys()), ["format", "path"], location, "reference_sequences")
        ParameterValidator.assert_in_valid_list(summary.upper(), [item.name for item in SequenceMatchingSummaryType], location, "summary")

        valid_formats = ReflectionHandler.discover_classes_by_partial_name("SequenceImport", "sequence_import/")
        ParameterValidator.assert_in_valid_list(f"{reference_sequences['format']}SequenceImport", valid_formats, location,
                                                "format in reference_sequences")

        assert os.path.isfile(reference_sequences["path"]), f"{location}: the file {reference_sequences['path']} does not exist. " \
                                                            f"Specify the correct path under reference_sequences."

        sequences = ReflectionHandler.get_class_by_name("{}SequenceImport".format(reference_sequences["format"]))\
            .import_items(reference_sequences["path"], paired=False)

        return {
            "max_edit_distance": max_edit_distance,
            "summary": SequenceMatchingSummaryType[summary.upper()],
            "reference_sequences": sequences
        }

    @staticmethod
    def build_object(dataset=None, **params):
        try:
            prepared_parameters = MatchedReferenceEncoder._prepare_parameters(**params)
            encoder = ReflectionHandler.get_class_by_name(MatchedReferenceEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "reference_encoding/")(**prepared_parameters)
        except ValueError:
            raise ValueError("{} is not defined for dataset of type {}.".format(MatchedReferenceEncoder.__name__,
                                                                                dataset.__class__.__name__))
        return encoder

    def encode(self, dataset, params: EncoderParams):

        cache_key = CacheHandler.generate_cache_key(self._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: self._encode_new_dataset(dataset, params))

        return encoded_dataset

    def _prepare_caching_params(self, dataset, params: EncoderParams):

        encoding_params_desc = {"max_edit_distance": self.max_edit_distance,
                                "summary": self.summary,
                                "reference_sequences": sorted([seq.get_sequence() + seq.metadata.v_gene + seq.metadata.j_gene
                                                               for seq in self.reference_sequences])}

        return (("dataset_identifiers", tuple(dataset.get_example_ids())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", MatchedReferenceEncoder.__name__),
                ("learn_model", params.learn_model),
                ("encoding_params", encoding_params_desc), )

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass
