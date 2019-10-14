import abc

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.util.ReflectionHandler import ReflectionHandler


class MatchedReceptorsEncoder(DatasetEncoder):
    """
    Encodes the dataset based on the matches between a dataset containing unpaired data (alpha/beta chains),
    and a paired reference receptor dataset.
    For each paired reference receptor, the frequency of either chain in the dataset is counted.

    Parameters:
        - list of reference sequences (TCABReceptor objects)
        - boolean value specifying whether one file represents one donor.
          If False, the metadata label "donor" must be specified for the datasets,
          this will be used to aggregate the frequency values.
    """

    dataset_mapping = {
        "RepertoireDataset": "MatchedReceptorsRepertoireEncoder"
    }

    # TODO: when adding arguments here also add them to _prepare_caching_params
    def __init__(self, reference_sequences: list, one_file_per_donor: bool):
        self.reference_sequences = reference_sequences
        self.one_file_per_donor = one_file_per_donor

    @staticmethod
    def create_encoder(dataset=None, params: dict = None):
        try:
            encoder = ReflectionHandler.get_class_by_name(
                MatchedReceptorsEncoder.dataset_mapping[dataset.__class__.__name__],
                "reference_encoding/")(**params if params is not None else {})
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

        chains = [((receptor.get_chain(receptor.get_chains()[0]), receptor.get_chain(receptor.get_chains()[1]))) for receptor in self.reference_sequences]

        encoding_params_desc = {"reference_sequences": sorted([chain_a.get_sequence() + chain_a.metadata.v_gene + chain_a.metadata.j_gene + "|" +
                                                                chain_b.get_sequence() + chain_b.metadata.v_gene + chain_b.metadata.j_gene
                                                                for chain_a, chain_b in chains]),
                                "one_file_per_donor": self.one_file_per_donor}

        return (("dataset_filenames", tuple(dataset.get_filenames())),
                ("dataset_metadata", dataset.metadata_file),
                ("dataset_type", dataset.__class__.__name__),
                ("labels", tuple(params["label_configuration"].get_labels_by_name())),
                ("encoding", MatchedReceptorsEncoder.__name__),
                ("learn_model", params["learn_model"]),
                ("encoding_params", encoding_params_desc), )


    @abc.abstractmethod
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        pass


    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])

