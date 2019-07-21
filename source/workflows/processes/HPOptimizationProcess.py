from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPOptimizationStrategy import HPOptimizationStrategy
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.hyperparameter_optimization.SplitConfig import SplitConfig
from source.ml_methods.MLMethod import MLMethod
from source.util.PathBuilder import PathBuilder
from source.workflows.processes.MLProcess import MLProcess
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams
from source.workflows.steps.DataSplitter import DataSplitter
from source.workflows.steps.DataSplitterParams import DataSplitterParams
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodAssessmentParams import MLMethodAssessmentParams
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams


class HPOptimizationProcess:
    """
    Class implementing hyper-parameter optimization and nested model training and assessment:

    The process is defined by two loops:
        - the outer loop over defined splits of the dataset for performance assessment
        - the inner loop over defined hyper-parameter space and with cross-validation or train & validation split
          to choose the best hyper-parameters

    Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

    """

    def __init__(self, dataset: Dataset, hp_strategy: HPOptimizationStrategy, path: str,
                 assessment: SplitConfig, selection: SplitConfig, metrics: list,
                 label_configuration: LabelConfiguration):
        self.dataset = dataset
        self.selection = selection
        self.hp_strategy = hp_strategy
        self.path = path
        self.batch_size = 10
        self.label_configuration = label_configuration
        self.metrics = metrics
        self.assessment = assessment

    def run_outer_cv(self):
        train_datasets, test_datasets = self.split_data(self.dataset, self.assessment)
        fold_performances = []
        for index in range(self.assessment.split_count):
            fold_performances.append(self.run_outer_fold(train_datasets[index], test_datasets[index], index+1))
        return sum(fold_performances)/len(fold_performances)

    def run_outer_fold(self, train_dataset, test_dataset, run):
        current_path = "{}{}/run_{}/".format(self.path, self.assessment.split_strategy.name, run)
        optimal_hp_setting = self.run_inner_cv(train_dataset)
        encoded_train_dataset = self.encode_dataset(train_dataset, optimal_hp_setting, current_path, learn_model=True)
        optimal_method = self.train_optimal_method(encoded_train_dataset, optimal_hp_setting, current_path)
        encoded_test_dataset = self.encode_dataset(test_dataset, optimal_hp_setting, current_path, learn_model=False)
        performance = MLMethodAssessment.run(MLMethodAssessmentParams(
            method=optimal_method,
            dataset=encoded_test_dataset,
            predictions_path="{}{}/run_{}/{}/predictions.csv".format(self.path, self.assessment.split_strategy.name, run, optimal_hp_setting),
            all_predictions_path="{}{}/all_predictions.csv".format(self.path, self.assessment.split_strategy.name),
            ml_details_path="{}{}/ml_details.csv".format(self.path, self.assessment.split_strategy.name),
            run=run,
            label_configuration=self.label_configuration,
            metrics=self.metrics,
            path=current_path
        ))
        return performance

    def run_inner_cv(self, train_dataset) -> HPSetting:
        train_datasets, test_datasets = self.split_data(train_dataset, self.selection)
        performance = -1
        hp_setting = None
        for hp_setting in self.hp_strategy.get_next_setting(hp_setting, performance):
            performance = self.test_hp_setting(hp_setting, train_datasets, test_datasets)

        return self.hp_strategy.get_optimal_hps()

    def train_optimal_method(self, dataset: Dataset, hp_setting: HPSetting, path: str) -> MLMethod:
        method = MLMethodTrainer.run(MLMethodTrainerParams(
            method=hp_setting.ml_method,
            result_path=path + "/ml_method/",
            dataset=dataset,
            labels=self.label_configuration.get_labels_by_name(),
            model_selection_cv=hp_setting.ml_params["model_selection_cv"],
            model_selection_n_folds=hp_setting.ml_params["_n_folds"],
            cores_for_training=-1  # TODO: make it configurable, add cores_for_training
        ))
        return method

    def encode_dataset(self, dataset: Dataset, hp_setting: HPSetting, path: str, learn_model: bool) -> Dataset:
        encoded_dataset = DataEncoder.run(DataEncoderParams(
            dataset=dataset,
            encoder=hp_setting.encoder,
            encoder_params=EncoderParams(
                model=hp_setting.encoder_params,
                result_path=path,
                batch_size=self.batch_size,
                label_configuration=self.label_configuration,
                learn_model=learn_model,
                filename="train_dataset.pkl" if learn_model else "test_dataset.pkl"
            )
        ))
        return encoded_dataset

    def test_hp_setting(self, hp_setting, train_datasets, test_datasets) -> float:
        fold_performances = []
        for index in range(self.selection.split_count):
            fold_performances.append(self.run_setting(hp_setting, train_datasets[index], test_datasets[index], index+1))
        return sum(fold_performances)/self.selection.split_count

    def run_setting(self, hp_setting, train_dataset, test_dataset, run_id: int):
        path = self.path + "{}/fold_{}/".format(hp_setting, run_id)
        PathBuilder.build(path)
        ml_process = MLProcess(train_dataset=train_dataset, test_dataset=test_dataset,
                               label_configuration=self.label_configuration, encoder=hp_setting.encoder,
                               encoder_params=hp_setting.encoder_params, method=hp_setting.ml_method,
                               ml_params=hp_setting.ml_params, metrics=self.metrics, path=path)
        return ml_process.run(run_id)

    def split_data(self, dataset, split_config: SplitConfig) -> tuple:
        params = DataSplitterParams(
            dataset=dataset,
            split_strategy=split_config.split_strategy,
            split_count=split_config.split_count,
            training_percentage=split_config.training_percentage,
            label_to_balance=split_config.label_to_balance
        )
        return DataSplitter.run(params)
