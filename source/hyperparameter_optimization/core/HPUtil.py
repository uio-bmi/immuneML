import copy

from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams
from source.workflows.steps.DataSplitter import DataSplitter
from source.workflows.steps.DataSplitterParams import DataSplitterParams
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class HPUtil:

    @staticmethod
    def split_data(dataset: Dataset, split_config: SplitConfig, path: str) -> tuple:
        paths = [f"{path}split_{i+1}/" for i in range(split_config.split_count)]
        params = DataSplitterParams(
            dataset=dataset,
            split_strategy=split_config.split_strategy,
            split_count=split_config.split_count,
            training_percentage=split_config.training_percentage,
            label_to_balance=split_config.label_to_balance,
            paths=paths
        )
        return DataSplitter.run(params)

    @staticmethod
    def get_average_performance(performances):
        if performances is not None and isinstance(performances, list) and len(performances) > 0:
            return sum(perf for perf in performances) / len(performances)
        else:
            return -1

    @staticmethod
    def preprocess_dataset(dataset: Dataset, preproc_sequence: list, path: str) -> Dataset:
        PathBuilder.build(path)
        tmp_dataset = dataset.clone()
        for preprocessing in preproc_sequence:
            tmp_dataset = preprocessing.process_dataset(tmp_dataset, path)
        return tmp_dataset

    @staticmethod
    def train_method(label: str, dataset, hp_setting: HPSetting, path: str, train_predictions_path, ml_details_path) -> MLMethod:
        method = MLMethodTrainer.run(MLMethodTrainerParams(
            method=copy.deepcopy(hp_setting.ml_method),
            result_path=path + "/ml_method/",
            dataset=dataset,
            label=label,
            train_predictions_path=train_predictions_path,
            ml_details_path=ml_details_path,
            model_selection_cv=hp_setting.ml_params["model_selection_cv"],
            model_selection_n_folds=hp_setting.ml_params["model_selection_n_folds"],
            cores_for_training=-1  # TODO: make it configurable, add cores_for_training
        ))
        return method

    @staticmethod
    def encode_dataset(dataset, hp_setting: HPSetting, path: str, learn_model: bool, context: dict, batch_size: int,
                       label_configuration: LabelConfiguration):
        PathBuilder.build(path)

        encoder = hp_setting.encoder.create_encoder(dataset, hp_setting.encoder_params).set_context(context)

        encoded_dataset = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=encoder,
            encoder_params=EncoderParams(
                model=hp_setting.encoder_params,
                result_path=path,
                batch_size=batch_size,
                label_configuration=label_configuration,
                learn_model=learn_model,
                filename="train_dataset.pkl" if learn_model else "test_dataset.pkl"
            )
        ))
        return encoded_dataset

    @staticmethod
    def assess_performance(state: HPOptimizationState, dataset, split_index, current_path, hp_setting, test_predictions_path: str,
                           label: str, ml_score_path: str):
        return MLMethodAssessment.run(MLMethodAssessmentParams(
            method=state.assessment_states[split_index].label_states[label].assessment_items[hp_setting].method,
            dataset=dataset,
            predictions_path=test_predictions_path,
            split_index=split_index,
            label=label,
            metrics=state.metrics,
            path=current_path,
            ml_score_path=ml_score_path
        ))
