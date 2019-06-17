import os

import numpy as np

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.util.FilenameHandler import FilenameHandler


class MatchedReferenceEncoder(DatasetEncoder):

    @staticmethod
    def encode(dataset: Dataset, params: EncoderParams) -> Dataset:

        filepath = params["result_path"] + FilenameHandler.get_dataset_name(MatchedReferenceEncoder.__name__)

        if os.path.isfile(filepath):
            encoded_dataset = PickleLoader.load(filepath)
        else:
            encoded_dataset = MatchedReferenceEncoder._encode_new_dataset(dataset, params)

        return encoded_dataset

    @staticmethod
    def _encode_new_dataset(dataset: Dataset, params: EncoderParams) -> Dataset:

        matched_info = MatchedReferenceEncoder._match_repertories(dataset, params)

        encoded_dataset = Dataset(filenames=dataset.get_filenames(), params=dataset.params,
                                  metadata_path=dataset.metadata_path)
        encoded_repertoires, labels = MatchedReferenceEncoder._encode_repertoires(dataset, matched_info, params)

        feature_name = params["model"]["summary"].name.lower()

        encoded_dataset.add_encoded_data(EncodedData(
            repertoires=encoded_repertoires,
            labels=labels,
            feature_names=[feature_name],
            repertoire_ids=[repertoire.identifier for repertoire in dataset.get_data()]
        ))

        MatchedReferenceEncoder.store(encoded_dataset, params)
        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: Dataset, matched_info, params: EncoderParams):
        encoded_repertories = np.zeros((dataset.get_repertoire_count(), 1), dtype=float)
        labels = {label: [] for label in params["label_configuration"].get_labels_by_name()}

        for index, repertoire in enumerate(dataset.get_data()):
            assert repertoire.identifier == matched_info["repertoires"][index]["repertoire"], \
                "MatchedReferenceEncoder: error in SequenceMatcher ordering of repertoires."
            encoded_repertories[index] = matched_info["repertoires"][index][params["model"]["summary"].name.lower()]
            for label_index, label in enumerate(params["label_configuration"].get_labels_by_name()):
                labels[label].append(repertoire.metadata.custom_params[label])

        return np.reshape(encoded_repertories, newshape=(-1, 1)), labels

    @staticmethod
    def _match_repertories(dataset: Dataset, params: EncoderParams):
        matcher = SequenceMatcher()
        matched_info = matcher.match(dataset=dataset,
                                     reference_sequences=params["model"]["reference_sequences"],
                                     max_distance=params["model"]["max_distance"],
                                     summary_type=params["model"]["summary"])
        return matched_info

    @staticmethod
    def store(encoded_dataset: Dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
