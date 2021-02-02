from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.ml_model_application.MLApplicationState import MLApplicationState


class MLApplicationInstruction(Instruction):
    """
    Instruction which enables using trained ML models and encoders on new datasets which do not necessarily have labeled data.

    The predictions are stored in the predictions.csv in the result path in the following format:


    .. list-table::
        :widths: 25 25 25 25
        :header-rows: 1

        * - example_id
          - cmv
          - cmv_true_proba
          - cmv_false_proba
        * - e1
          - True
          - 0.8
          - 0.2
        * - e2
          - False
          - 0.2
          - 0.8
        * - e3
          - True
          - 0.78
          - 0.22

    Arguments:

        dataset: dataset for which examples need to be classified

        config_path: path to the zip file exported from MLModelTraining instruction (which includes train ML model, encoder, preprocessing etc.)

        number_of_processes (int): number of processes to use for prediction

        store_encoded_data (bool): whether encoded dataset should be stored on disk; can be True or False; setting this argument to True might
        increase the disk space usage

    Specification example for the MLApplication instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            number_of_processes: 4
            label: CD
            store_encoded_data: False

    """

    def __init__(self, dataset: Dataset, label_configuration: LabelConfiguration, hp_setting: HPSetting, number_of_processes: int, name: str,
                 store_encoded_data: bool):

        self.state = MLApplicationState(dataset=dataset, hp_setting=hp_setting, label_config=label_configuration, pool_size=number_of_processes, name=name,
                                        store_encoded_data=store_encoded_data)

    def run(self, result_path: Path):
        self.state.path = PathBuilder.build(result_path / self.state.name)

        dataset = self.state.dataset

        if self.state.hp_setting.preproc_sequence is not None:
            dataset = HPUtil.preprocess_dataset(dataset, self.state.hp_setting.preproc_sequence, self.state.path)

        dataset = HPUtil.encode_dataset(dataset, self.state.hp_setting, self.state.path, learn_model=False, number_of_processes=self.state.pool_size,
                                        label_configuration=self.state.label_config, context={}, encode_labels=False,
                                        store_encoded_data=self.state.store_encoded_data)

        self._make_predictions(dataset)

        return self.state

    def _make_predictions(self, dataset):

        label = self.state.label_config.get_labels_by_name()[0]

        method = self.state.hp_setting.ml_method
        predictions = method.predict(dataset.encoded_data, label)
        predictions_df = pd.DataFrame({"example_id": dataset.get_example_ids(), label: predictions[label]})

        if type(dataset) == RepertoireDataset:
            predictions_df.insert(0, 'repertoire_file', [repertoire.data_filename.name for repertoire in dataset.get_data()])

        if method.can_predict_proba():
            classes = method.get_classes_for_label(label)
            predictions_proba = method.predict_proba(dataset.encoded_data, label)[label]
            for cls_index, cls in enumerate(classes):
                predictions_df[f'{label}_{cls}_proba'] = predictions_proba[:, cls_index]

        self.state.predictions_path = self.state.path / "predictions.csv"
        predictions_df.to_csv(self.state.predictions_path, index=False)
