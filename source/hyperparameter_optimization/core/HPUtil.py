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
        params = DataSplitterParams(
            dataset=dataset,
            split_strategy=split_config.split_strategy,
            split_count=split_config.split_count,
            training_percentage=split_config.training_percentage,
            label_to_balance=split_config.label_to_balance,
            path=path
        )
        return DataSplitter.run(params)

    @staticmethod
    def get_average_performance(metrics_per_label, label):
        if all(label in performance for performance in metrics_per_label):
            return sum(perf[label] for perf in metrics_per_label) / len(metrics_per_label)
        else:
            return metrics_per_label

    @staticmethod
    def preprocess_dataset(dataset: Dataset, preproc_sequence: list, path: str) -> Dataset:
        PathBuilder.build(path)
        tmp_dataset = dataset.clone()
        for preprocessing in preproc_sequence:
            tmp_dataset = preprocessing.process_dataset(tmp_dataset, path)
        return tmp_dataset

    @staticmethod
    def train_method(state: HPOptimizationState, dataset, hp_setting: HPSetting, path: str) -> MLMethod:
        method = MLMethodTrainer.run(MLMethodTrainerParams(
            method=copy.deepcopy(hp_setting.ml_method),
            result_path=path + "/ml_method/",
            dataset=dataset,
            labels=state.label_configuration.get_labels_by_name(),
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
    def assess_performance(state: HPOptimizationState, dataset, run, current_path, hp_setting):
        method_per_label = {label: state.assessment_states[run].label_states[label].assessment_items[hp_setting].method
                            for label in state.assessment_states[run].label_states}
        return MLMethodAssessment.run(MLMethodAssessmentParams(
            method=method_per_label,
            dataset=dataset,
            predictions_path="{}predictions.csv".format(current_path),
            all_predictions_path="{}assessment_{}/all_predictions.csv".format(state.path, state.assessment_config.split_strategy.name),
            ml_details_path="{}ml_details.csv".format(current_path),
            run=run,
            label_configuration=state.label_configuration,
            metrics=state.metrics,
            path=current_path
        ))
