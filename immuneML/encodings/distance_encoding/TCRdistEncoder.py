import pandas as pd

from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, ElementDataset, SequenceDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.Logger import print_log


class TCRdistEncoder(DatasetEncoder):
    """
    Encodes the given ReceptorDataset as a distance matrix between all receptors, where the distance is computed using TCRdist from the paper:
    Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires.
    Nature. 2017; 547(7661):89-93. `doi:10.1038/nature22383 <https://www.nature.com/articles/nature22383>`_.

    For the implementation, `TCRdist3 <https://tcrdist3.readthedocs.io/en/latest/>`_ library was used (source code available
    `here <https://github.com/kmayerb/tcrdist3>`_).

    **Dataset type:**

    - ReceptorDataset

    - SequenceDataset


    **Specification arguments:**

    - cores (int): number of processes to use for the computation


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_tcr_dist_enc:
                    TCRdist:
                        cores: 4

    """

    def __init__(self, cores: int, name: str = None):
        super().__init__(name=name)
        self.cores = cores
        self.distance_matrix = None
        self.context = None
        self._tmp_results_path = None

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, ReceptorDataset) or isinstance(dataset, SequenceDataset):
            return TCRdistEncoder(**params)
        else:
            raise ValueError("TCRdistEncoder is defined for receptor and sequence dataset.")

    def set_context(self, context: dict):
        self.context = context
        return self

    def encode(self, dataset, params: EncoderParams):
        if self._tmp_results_path is None and params.learn_model:
            self._tmp_results_path = params.result_path

        train_receptor_ids = EncoderHelper.prepare_training_ids(dataset, params, self._tmp_results_path)

        if params.learn_model:
            self._build_tcr_dist_matrix(dataset, params.label_config.get_labels_by_name())

        distance_matrix = self.distance_matrix.loc[dataset.get_example_ids(), train_receptor_ids]
        labels = self._build_labels(dataset, params) if params.encode_labels else None

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix.values, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=TCRdistEncoder.__name__)
        return encoded_dataset

    def _build_tcr_dist_matrix(self, dataset: ElementDataset, label_names):
        from immuneML.util.TCRdistHelper import TCRdistHelper

        chains = TCRdistHelper.get_chains(dataset)

        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]
        tcr_rep = TCRdistHelper.compute_tcr_dist(current_dataset, label_names, self.cores)

        data = 0.

        if 'alpha' in chains:
            data += tcr_rep.pw_alpha
        if 'beta' in chains:
            data += tcr_rep.pw_beta

        self.distance_matrix = pd.DataFrame(data, index=tcr_rep.clone_df.clone_id.values,
                                            columns=tcr_rep.clone_df.clone_id.values)

    def _build_labels(self, dataset: ElementDataset, params: EncoderParams) -> dict:
        return dataset.get_metadata(params.label_config.get_labels_by_name())

