from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringItem


@dataclass
class ValidateClusteringState:
    cl_item: ClusteringItem = None
    dataset: Dataset = None
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
                validation_type: [method_based, result_based]

    """

    def __init__(self, clustering_item: ClusteringItem, dataset: Dataset, validation_type: List[str],
                 result_path: Path = None):
        self._state = ValidateClusteringState(clustering_item, dataset, validation_type, result_path)

    def run(self, result_path: Path) -> ValidateClusteringState:
        return self._state