import os

import numpy as np

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.Dataset import Dataset
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

        encoded_dataset = Dataset(filenames=dataset.filenames, params=dataset.params)
        encoded_repertoires, labels = MatchedReferenceEncoder._encode_repertoires(dataset, matched_info, params)

        feature_name = "percentage_of_sequences_matched" \
            if "percentages" in params["model"] and params["model"]["percentages"] \
            else "count_of_sequences_matched"

        encoded_dataset.encoded_data = {
            "repertoires": encoded_repertoires,
            "labels": labels,
            "label_names": params["label_configuration"].get_labels_by_name(),
            "feature_names": [feature_name]
        }

        MatchedReferenceEncoder.store(encoded_dataset, params)
        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: Dataset, matched_info, params: EncoderParams):
        encoded_repertories = np.zeros((dataset.get_repertoire_count(), 1), dtype=float)
        labels = np.zeros(shape=(params["label_configuration"].get_label_count(), dataset.get_repertoire_count()))
        c = 100 if "percentages" in params["model"] and params["model"]["percentages"] else 1

        for index, repertoire in enumerate(dataset.get_data()):
            assert index == matched_info["repertoires"][index]["repertoire_index"], \
                "MatchedReferenceEncoder: error in SequenceMatcher ordering of repertoires."
            encoded_repertories[index] = matched_info["repertoires"][index]["percentage_of_sequences_matched"] * c
            for label_index, label in enumerate(params["label_configuration"].get_labels_by_name()):
                labels[label_index][index] = repertoire.metadata.custom_params[label]

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
        PickleExporter.export(encoded_dataset, params["result_path"],
                              FilenameHandler.get_dataset_name(MatchedReferenceEncoder.__name__))
