from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.EncoderHelper import EncoderHelper


class TCRdistEncoder(DatasetEncoder):
    """
    Encodes the given ReceptorDataset as a distance matrix between all receptors, where the distance is computed using TCRdist from the paper:
    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_.

    For the implementation, `TCRdist3 <https://tcrdist3.readthedocs.io/en/latest/>`_ library was used (source code available
    `here <https://github.com/kmayerb/tcrdist3>`_).

    Arguments:

        cores (int): number of processes to use for the computation

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_tcr_dist_enc: # user-defined name
            TCRdist:
                cores: 4

    """

    def __init__(self, cores: int, name: str = None):
        self.cores = cores
        self.name = name
        self.distance_matrix = None
        self.context = None

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, ReceptorDataset):
            return TCRdistEncoder(**params)
        else:
            raise ValueError("TCRdistEncoder is not defined for dataset types which are not ReceptorDataset.")

    def set_context(self, context: dict):
        self.context = context
        return self

    def encode(self, dataset, params: EncoderParams):
        train_receptor_ids = EncoderHelper.prepare_training_ids(dataset, params)
        if params.learn_model:
            self._build_tcr_dist_matrix(dataset, params.label_config.get_labels_by_name())

        distance_matrix = self.distance_matrix.loc[dataset.get_example_ids(), train_receptor_ids]
        labels = self._build_labels(dataset, params) if params.encode_labels else None

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=TCRdistEncoder.__name__)

        return encoded_dataset

    def _build_tcr_dist_matrix(self, dataset: ReceptorDataset, labels):
        from immuneML.util.TCRdistHelper import TCRdistHelper

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]
        tcr_rep = TCRdistHelper.compute_tcr_dist(current_dataset, labels, self.cores)
        self.distance_matrix = pd.DataFrame(tcr_rep.pw_alpha + tcr_rep.pw_beta, index=tcr_rep.clone_df.clone_id.values,
                                            columns=tcr_rep.clone_df.clone_id.values)

    def _build_labels(self, dataset: ReceptorDataset, params: EncoderParams) -> dict:
        labels = {label: [] for label in params.label_config.get_labels_by_name()}
        for receptor in dataset.get_data():
            for label in labels.keys():
                labels[label].append(receptor.metadata[label])
        return labels

    @staticmethod
    def export_encoder(path: Path, encoder) -> str:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file
