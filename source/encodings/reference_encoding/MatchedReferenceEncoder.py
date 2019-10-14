import abc

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReferenceEncoder(DatasetEncoder):

    dataset_mapping = {
        "RepertoireDataset": "ReferenceRepertoireEncoder"
    }

    def __init__(self, max_edit_distance: int, summary: SequenceMatchingSummaryType, reference_sequences: list):
        self.max_edit_distance = max_edit_distance
        self.summary = summary
        self.reference_sequences = reference_sequences

    @staticmethod
    def create_encoder(dataset=None, params: dict = None):
        try:
            encoder = ReflectionHandler.get_class_by_name(MatchedReferenceEncoder.dataset_mapping[dataset.__class__.__name__],
                                                          "reference_encoding/")(**params if params is not None else {})
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

        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("encoding", MatchedReferenceEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("encoding_params", encoding_params_desc), )

    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
