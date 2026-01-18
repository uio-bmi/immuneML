import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering import clustering_runner
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


@dataclass
class ValidateClusteringState:
    cl_item: ClusteringItem = None
    dataset: Dataset = None
    metrics: List[str] = None
    validation_type: List[str] = None
    result_path: Path = None
    name: str = "validate_clustering"
    label_config: LabelConfiguration = None
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    region_type: RegionType = RegionType.IMGT_CDR3
    number_of_processes: int = 1
    method_based_result: ClusteringItem = None
    result_based_result: ClusteringItem = None
    method_based_predictions_path: Path = None
    result_based_predictions_path: Path = None
    method_based_report_results: List[ReportResult] = field(default_factory=list)
    result_based_report_results: List[ReportResult] = field(default_factory=list)


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
                clustering_config_path: /path/to/exported_clustering.zip
                dataset: val_dataset
                metrics: [adjusted_rand_score, silhouette_score]
                validation_type: [method_based, result_based]

    """

    def __init__(self, clustering_item: ClusteringItem, dataset: Dataset, metrics: List[str], validation_type: List[str],
                 label_config: LabelConfiguration = None, sequence_type: SequenceType = SequenceType.AMINO_ACID,
                 region_type: RegionType = RegionType.IMGT_CDR3, number_of_processes: int = 1,
                 result_path: Path = None):
        self._state = ValidateClusteringState(
            cl_item=clustering_item,
            dataset=dataset,
            metrics=metrics,
            validation_type=validation_type,
            result_path=result_path,
            label_config=label_config if label_config else LabelConfiguration(),
            sequence_type=sequence_type,
            region_type=region_type,
            number_of_processes=number_of_processes
        )

    def run(self, result_path: Path) -> ValidateClusteringState:
        self._state.result_path = PathBuilder.build(result_path)

        print_log(f"ValidateClusteringInstruction: starting validation with types {self._state.validation_type}")

        # Initialize predictions DataFrame with example IDs and labels
        predictions_df = self._init_predictions_df()

        if 'method_based' in self._state.validation_type:
            predictions_df = self._run_method_based_validation(predictions_df)

        if 'result_based' in self._state.validation_type:
            predictions_df = self._run_result_based_validation(predictions_df)

        print_log("ValidateClusteringInstruction: validation completed.")

        return self._state

    def _init_predictions_df(self) -> pd.DataFrame:
        """Initialize predictions DataFrame with labels and example IDs."""
        if self._state.label_config and len(self._state.label_config.get_labels_by_name()) > 0:
            predictions_df = self._state.dataset.get_metadata(
                self._state.label_config.get_labels_by_name(), return_df=True
            )
        else:
            predictions_df = pd.DataFrame(index=range(self._state.dataset.get_example_count()))

        predictions_df['example_id'] = self._state.dataset.get_example_ids()
        return predictions_df

    def _run_method_based_validation(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run method-based validation: refit the clustering setting on the new dataset.
        Uses fresh copies of encoder and clustering method to refit from scratch.
        """
        print_log("ValidateClusteringInstruction: running method-based validation")
        path = PathBuilder.build(self._state.result_path / "method_based_validation")

        cl_setting = copy.deepcopy(self._state.cl_item.cl_setting)
        cl_setting.path = path

        # Run clustering with learn_model=True (refit encoder and clustering)
        cl_item_result, predictions_df = clustering_runner.run_setting(
            dataset=self._state.dataset,
            cl_setting=cl_setting,
            path=path,
            predictions_df=predictions_df,
            metrics=self._state.metrics,
            label_config=self._state.label_config,
            number_of_processes=self._state.number_of_processes,
            sequence_type=self._state.sequence_type,
            region_type=self._state.region_type,
            evaluate=True
        )

        self._state.method_based_result = cl_item_result.item
        self._state.method_based_predictions_path = path / "method_based_predictions.csv"
        predictions_df.to_csv(self._state.method_based_predictions_path, index=False)

        print_log(f"ValidateClusteringInstruction: method-based validation completed. "
                  f"Predictions saved to {self._state.method_based_predictions_path}")

        return predictions_df

    def _run_result_based_validation(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run result-based validation: use the classifier to predict clusters on the new dataset.
        Encoding is applied with learn_model=False to use the same transformation as discovery.
        """
        print_log("ValidateClusteringInstruction: running result-based validation")
        path = PathBuilder.build(self._state.result_path / "result_based_validation")

        cl_setting = self._state.cl_item.cl_setting
        cl_setting.path = path
        classifier = self._state.cl_item.classifier

        if classifier is None:
            logging.warning("ValidateClusteringInstruction: No classifier available for result-based validation. "
                            "Skipping result-based validation.")
            return predictions_df

        # Apply the trained classifier to get cluster predictions
        cl_item = clustering_runner.apply_cluster_classifier(
            dataset=self._state.dataset,
            cl_setting=cl_setting,
            classifier=classifier,
            encoder=self._state.cl_item.encoder,
            predictions_path=path / "result_based_predictions.csv",
            number_of_processes=self._state.number_of_processes,
            sequence_type=self._state.sequence_type,
            region_type=self._state.region_type
        )

        # Add predictions to DataFrame
        predictions_df[f'predictions_result_based_{cl_setting.get_key()}'] = cl_item.predictions

        # Evaluate clustering metrics
        features = clustering_runner.get_features(cl_item.dataset, cl_setting)
        cl_item = clustering_runner.evaluate_clustering(
            predictions_df=predictions_df,
            cl_setting=cl_setting,
            features=features,
            metrics=self._state.metrics,
            label_config=self._state.label_config,
            cl_item=cl_item
        )

        self._state.result_based_result = cl_item
        self._state.result_based_predictions_path = path / "result_based_predictions.csv"

        print_log(f"ValidateClusteringInstruction: result-based validation completed. "
                  f"Predictions saved to {self._state.result_based_predictions_path}")

        return predictions_df