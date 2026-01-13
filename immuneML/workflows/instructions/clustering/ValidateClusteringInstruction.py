from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringItem, ClusteringConfig
from immuneML.workflows.instructions.clustering.ValidationHandler import ValidationHandler


@dataclass
class ValidateClusteringState:
    cl_item: ClusteringItem = None
    dataset: Dataset = None
    metrics: List[str] = None
    validation_type: List[str] = None
    result_path: Path = None


class ValidateClusteringInstruction(Instruction):
    """
    ValidateClustering instruction supports the application of the chosen clustering setting (preprocessing, encoding,
    clustering, with all hyperparameters) to a new dataset for validation.

    For more details on validating the clustering algorithm and its hyperparameters, see the paper:
    Ullmann, T., Hennig, C., & Boulesteix, A.-L. (2022). Validation of cluster analysis results on validation
    data: A systematic framework. WIREs Data Mining and Knowledge Discovery, 12(3), e1444.
    https://doi.org/10.1002/widm.1444

    **Specification arguments:**

    - clustering_config_path (str): path to the clustering exported by the Clustering instruction that will be applied
      to the new dataset

    - dataset (str): name of the validation dataset to which the clustering will be applied, as defined under definitions

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
      completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
      clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

    - validation_type (list): how to perform validation; options are `method_based` validation (refit the clustering
      algorithm to the new dataset and compare the clusterings) and `result_based` validation (transfer the clustering
      from the original dataset to the validation dataset using a supervised classifier and compare the clusterings)


    **YAML specification:**

    .. code-block:: yaml

        instructions:
            validate_clustering_inst:
                type: ValidateClustering
                clustering_config_path: /path/to/clustering_config.yaml
                dataset: val_dataset
                metrics: [adjusted_rand_score, silhouette_score]
                validation_type: [method_based, result_based]

    """

    def __init__(self, clustering_item: ClusteringItem, dataset: Dataset, metrics: List[str], validation_type: List[str],
                 result_path: Path = None):
        self._state = ValidateClusteringState(clustering_item, dataset, metrics, validation_type, result_path)

    def run(self, result_path: Path) -> ValidateClusteringState:

        # ValidationHandler(ClusteringConfig(self._state.cl_item.cl_setting.get_key(), self._state.dataset, self._state.metrics, )

        return self._state