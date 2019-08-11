import numpy as np

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.analysis.SequenceMatcher import SequenceMatcher
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams


class MatchedReferenceEncoder(DatasetEncoder):

    @staticmethod
    def encode(dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        cache_key = CacheHandler.generate_cache_key(MatchedReferenceEncoder._prepare_caching_params(dataset, params))
        encoded_dataset = CacheHandler.memo(cache_key,
                                            lambda: MatchedReferenceEncoder._encode_new_dataset(dataset, params))

        return encoded_dataset

    @staticmethod
    def _prepare_caching_params(dataset: RepertoireDataset, params: EncoderParams):

        encoding_params_desc = {"max_distance": params["model"]["max_distance"],
                                "summary": params["model"]["summary"],
                                "reference_sequences": sorted([seq.get_sequence() + seq.metadata.v_gene + seq.metadata.j_gene
                                                        for seq in params["model"]["reference_sequences"]])}

        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("encoding", MatchedReferenceEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("encoding_params", encoding_params_desc), )

    @staticmethod
    def _encode_new_dataset(dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:

        matched_info = MatchedReferenceEncoder._match_repertories(dataset, params)

        encoded_dataset = RepertoireDataset(filenames=dataset.get_filenames(), params=dataset.params,
                                            metadata_file=dataset.metadata_file)
        encoded_repertoires, labels = MatchedReferenceEncoder._encode_repertoires(dataset, matched_info, params)

        feature_name = params["model"]["summary"].name.lower()

        encoded_dataset.add_encoded_data(EncodedData(
            examples=encoded_repertoires,
            labels=labels,
            feature_names=[feature_name],
            example_ids=[repertoire.identifier for repertoire in dataset.get_data()],
            encoding=MatchedReferenceEncoder.__name__
        ))

        MatchedReferenceEncoder.store(encoded_dataset, params)
        return encoded_dataset

    @staticmethod
    def _encode_repertoires(dataset: RepertoireDataset, matched_info, params: EncoderParams):
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
    def _match_repertories(dataset: RepertoireDataset, params: EncoderParams):
        matcher = SequenceMatcher()
        matched_info = matcher.match(dataset=dataset,
                                     reference_sequences=params["model"]["reference_sequences"],
                                     max_distance=params["model"]["max_distance"],
                                     summary_type=params["model"]["summary"])
        return matched_info

    @staticmethod
    def store(encoded_dataset: RepertoireDataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
