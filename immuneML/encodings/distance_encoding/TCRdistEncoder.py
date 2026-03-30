import pandas as pd

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, ElementDataset, SequenceDataset
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

    **Dataset type:**

    - ReceptorDataset

    - SequenceDataset


    **Specification arguments:**

    - cores (int): number of processes to use for the computation

    - cdr3_only (bool): whether to use only cdr3 or also v gene; if set to false, encoding will only compute the distances
      between the CDR3 regions of the receptors


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_tcr_dist_enc:
                    TCRdist:
                        cores: 4
                        cdr3_only: false # default tcrdist behavior

    """

    def __init__(self, cores: int, cdr3_only: bool, name: str = None):
        super().__init__(name=name)
        self.cores = cores
        self.cdr3_only = cdr3_only
        self.distance_matrix = None
        self.context = None
        self._tmp_results_path = None
        self.training_ids = None
        self.training_df = None
        self.training_chains = None
        self.organism = None

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

        if params.learn_model:
            train_receptor_ids = EncoderHelper.prepare_training_ids(dataset, params, self._tmp_results_path)
            self.training_ids = list(train_receptor_ids)
            self._build_tcr_dist_matrix(dataset, params.label_config.get_labels_by_name())
        else:
            train_receptor_ids = self.training_ids if self.training_ids is not None \
                else EncoderHelper.prepare_training_ids(dataset, params, self._tmp_results_path)
            self._extend_distance_matrix(dataset, params.label_config.get_labels_by_name())

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
        organism = self._get_organism(current_dataset)
        tcr_rep = TCRdistHelper.compute_tcr_dist(current_dataset, label_names, self.cores, self.cdr3_only,
                                                  organism=organism)

        self.organism = tcr_rep.organism
        self.training_df = tcr_rep.clone_df
        self.training_chains = chains

        data = 0.

        if 'alpha' in chains:
            data += tcr_rep.pw_alpha
        if 'beta' in chains:
            data += tcr_rep.pw_beta

        self.distance_matrix = pd.DataFrame(data, index=tcr_rep.clone_df.clone_id.values,
                                            columns=tcr_rep.clone_df.clone_id.values)

    def _get_organism(self, dataset: ElementDataset) -> str:
        labels = dataset.labels if isinstance(dataset.labels, dict) else {}
        org_val = labels.get("organism")
        if isinstance(org_val, str):
            return org_val
        if isinstance(org_val, list) and len(org_val) == 1:
            return org_val[0]
        return self.organism

    def _extend_distance_matrix(self, dataset: ElementDataset, label_names):
        """Compute cross-distances between new sequences and training sequences, extending the distance matrix."""
        from immuneML.util.TCRdistHelper import TCRdistHelper

        new_ids = [id_ for id_ in dataset.get_example_ids() if id_ not in self.distance_matrix.index]
        if not new_ids:
            return

        tcr_rep = TCRdistHelper.compute_tcr_dist_rect(dataset, self.training_df, self.training_chains,
                                                       self.organism, label_names, self.cores, self.cdr3_only)

        data = 0.
        if 'alpha' in self.training_chains:
            data += tcr_rep.rw_alpha
        if 'beta' in self.training_chains:
            data += tcr_rep.rw_beta

        cross_df = pd.DataFrame(data, index=tcr_rep.clone_df.clone_id.values,
                                columns=self.training_df.clone_id.values)
        new_rows = cross_df.loc[[id_ for id_ in cross_df.index if id_ not in self.distance_matrix.index]]
        self.distance_matrix = pd.concat([self.distance_matrix, new_rows])

    def _build_labels(self, dataset: ElementDataset, params: EncoderParams) -> dict:
        return dataset.get_metadata(params.label_config.get_labels_by_name())

