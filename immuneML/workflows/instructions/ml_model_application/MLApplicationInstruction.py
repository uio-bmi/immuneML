import warnings

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ml_model_application.MLApplicationState import MLApplicationState
from scripts.specification_util import update_docs_per_mapping


class MLApplicationInstruction(Instruction):
    """
    Instruction which enables using trained ML models and encoders on new datasets which do not necessarily have labeled data.
    When the same label is provided as the ML setting was trained for, performance metrics can be computed.

    The predictions are stored in the predictions.csv in the result path in the following format:

    .. list-table::
        :widths: 25 25 25 25
        :header-rows: 1

        * - example_id
          - cmv_predicted_class
          - cmv_1_proba
          - cmv_0_proba
        * - e1
          - 1
          - 0.8
          - 0.2
        * - e2
          - 0
          - 0.2
          - 0.8
        * - e3
          - 1
          - 0.78
          - 0.22


    If the same label that the ML setting was trained for is present in the provided dataset, the 'true' label value
    will be added to the predictions table in addition:

    .. list-table::
        :widths: 20 20 20 20 20
        :header-rows: 1

        * - example_id
          - cmv_predicted_class
          - cmv_1_proba
          - cmv_0_proba
          - cmv_true_class
        * - e1
          - 1
          - 0.8
          - 0.2
          - 1
        * - e2
          - 0
          - 0.2
          - 0.8
          - 0
        * - e3
          - 1
          - 0.78
          - 0.22
          - 0

    Specification arguments:

    - dataset: dataset for which examples need to be classified

    - config_path: path to the zip file exported from MLModelTraining instruction (which includes train ML model, encoder, preprocessing etc.)

    - number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

    - metrics (list): a list of metrics to compute between the true and predicted classes. These metrics will only be computed when the same label with the same classes is provided for the dataset as the original label the ML setting was trained for.


    Specification example for the MLApplication instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            metrics:
            - accuracy
            - precision
            - recall
            number_of_processes: 4

    """

    def __init__(self, dataset: Dataset, label_configuration: LabelConfiguration, hp_setting: HPSetting, metrics: List[ClassificationMetric], number_of_processes: int, name: str):

        self.state = MLApplicationState(dataset=dataset, hp_setting=hp_setting, label_config=label_configuration, metrics=metrics, pool_size=number_of_processes, name=name)

    def run(self, result_path: Path):
        self.state.path = PathBuilder.build(result_path / self.state.name)

        dataset = self.state.dataset

        if self.state.hp_setting.preproc_sequence is not None:
            dataset = HPUtil.preprocess_dataset(dataset, self.state.hp_setting.preproc_sequence, self.state.path)

        dataset = HPUtil.encode_dataset(dataset, self.state.hp_setting, self.state.path, learn_model=False, number_of_processes=self.state.pool_size,
                                        label_configuration=self.state.label_config, context={}, encode_labels=False)

        self._write_outputs(dataset)

        return self.state

    def _write_outputs(self, dataset):
        label = self.state.label_config.get_label_objects()[0]

        predictions_df = self._make_predictions_df(dataset, label)
        self.state.predictions_path = self.state.path / "predictions.csv"
        predictions_df.to_csv(self.state.predictions_path, index=False)

        metrics_df = self._apply_metrics_with_warnings(dataset, label, predictions_df)
        if metrics_df is not None:
            self.state.metrics_path = self.state.path / "metrics.csv"
            metrics_df.to_csv(self.state.metrics_path, index=False)


    def _make_predictions_df(self, dataset, label):
        method = self.state.hp_setting.ml_method
        predictions = method.predict(dataset.encoded_data, label)
        predictions_df = pd.DataFrame({"example_id": dataset.get_example_ids()})

        predictions_df[f"{label.name}_predicted_class"] = predictions[label.name]
        predictions_df[f"{label.name}_predicted_class"] = predictions_df[f"{label.name}_predicted_class"].astype(str)

        if type(dataset) == RepertoireDataset:
            predictions_df.insert(0, 'repertoire_file', [repertoire.data_filename.name for repertoire in dataset.get_data()])

        if method.can_predict_proba():
            predictions_proba = method.predict_proba(dataset.encoded_data, label)[label.name]

            for cls in method.get_classes():
                predictions_df[f'{label.name}_{cls}_proba'] = predictions_proba[cls]

        if label.name in dataset.get_label_names():
            predictions_df[f"{label.name}_true_class"] = dataset.get_metadata([label.name])[label.name]
            predictions_df[f"{label.name}_true_class"] = predictions_df[f"{label.name}_true_class"].astype(str)

        return predictions_df

    def _apply_metrics_with_warnings(self, dataset, label, predictions_df):
        if len(self.state.metrics) > 0:
            if label.name in dataset.get_label_names():
                if all([dataset_class in self.state.hp_setting.ml_method.get_classes() for dataset_class in dataset.labels[label.name]]):
                    return self._apply_metrics(label, predictions_df)
                else:
                    warnings.warn(f"MLApplicationInstruction: tried to apply metrics for label {label.name}. "
                                  f"Found class values {dataset.labels[label.name]} in the provided dataset, "
                                  f"but expected classes {self.state.hp_setting.ml_method.get_classes()}.")
            else:
                warnings.warn(f"MLApplicationInstruction: tried to apply metrics for label {label.name}, "
                              f"but the provided dataset only contains information for the following "
                              f"labels: {dataset.get_label_names()}.")

    def _apply_metrics(self, label, predictions_df):
        result = {}
        for metric in self.state.metrics:
            if all([f'{label.name}_{cls}_proba' in predictions_df.columns for cls in label.values]):
                predicted_proba_y = np.vstack([np.array(predictions_df[f'{label.name}_{cls}_proba']) for cls in label.values]).T
            else:
                predicted_proba_y = None

            result[metric.name.lower()] = [MetricUtil.score_for_metric(metric=metric,
                                                                        predicted_y=np.array(predictions_df[f"{label.name}_predicted_class"]),
                                                                        predicted_proba_y=predicted_proba_y,
                                                                        true_y=np.array(predictions_df[f"{label.name}_true_class"]),
                                                                        classes=[str(val) for val in label.values])]
        return pd.DataFrame(result)

    @staticmethod
    def get_documentation():
        doc = str(MLApplicationInstruction.__doc__)
        valid_metrics = str([metric.name.lower() for metric in ClassificationMetric])[1:-1].replace("'", "`")

        mapping = {
            "a list of metrics": f"a list of metrics ({valid_metrics})",
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
