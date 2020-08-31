import datetime
from typing import List, Set

from scripts.specification_util import update_docs_per_mapping
from source.data_model.dataset.Dataset import Dataset
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.reports.data_reports.DataReport import DataReport
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.reports.ml_reports.MLReport import MLReport
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.MLProcess import MLProcess
from source.workflows.instructions.ml_model_training.MLModelTrainingState import MLModelTrainingState


class MLModelTrainingInstruction(Instruction):
    """
    Instruction that trains the ML model on the full dataset and exports the model and the setting details. It is useful after the hyperparameter
    optimization where the optimal model has been chosen.

    Note that the performance measured in this instruction is the performance on the training dataset and as such is not a good indicator of the
    performance for the task, but rather shows how well the model is fitted to the dataset.

    Arguments:

        dataset: identifier of the dataset which will be used to train the ML model

        preprocessing: the preprocessing sequence that will be applied to the dataset before encoding the dataset and training the model; can be None

        encoding: the encoding that will be applied to the dataset after it is preprocessed if preprocessing was specified

        ml_model: the machine learning model to train on the dataset; trained model is the output of this instruction

        number_of_processes: how many processes to use while training the model

        metrics: a list of metrics that will be computed on the dataset (used for training), but which will not be used for fitting the model;

        optimization_metric: a metric that will be used for fitting the model

        labels: a list of labels to fit the model for; in the final output, there will be one model per label listed here

        reports: a list of reports too be executed on the dataset ('data'), encoded dataset ('encoding') and ML model ('models')

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        training_ML_instruction: # user-defined instruction name
            type: MLModelTraining
            dataset: dataset1 # dataset name as specified under 'definitions' part of the specification
            encoding: encoding1 # encoding name as specified under 'definitions' part of the specification
            preprocessing: seq1  # the name of the preprocessing sequence as specified under 'definitions' part of the specification
            ml_model: ml1  # ml model name as specified under 'definitions' part of the specification
            number_of_processes: 8
            metrics:  # a list of metric names to be computed on the dataset
                - accuracy
                - precision
                - recall
            optimization_metric: balanced_accuracy # the one metric that will be used for optimization
            labels: # list of labels
                - CMV
            reports: # reports to be executed on the dataset, encoded data and the trained model
                data:
                    - report1
                    - report2
                encoding:
                    - report3
                models:
                    - report4

    """

    def __init__(self, dataset: Dataset, metrics: Set[Metric], optimization_metric: Metric, ml_reports: List[MLReport],
                 encoding_reports: List[EncodingReport], data_reports: List[DataReport], number_of_processes: int, label_config: LabelConfiguration,
                 hp_setting: HPSetting, name: str = None):

        processes = []

        for label in label_config.get_labels_by_name():
            process = MLProcess(train_dataset=dataset, test_dataset=dataset, label=label, metrics=metrics, optimization_metric=optimization_metric,
                                path=None, ml_reports=ml_reports, encoding_reports=encoding_reports, data_reports=data_reports, hp_setting=hp_setting,
                                number_of_processes=number_of_processes, label_config=label_config, report_context={"dataset": dataset})

            processes.append(process)

        self.state = MLModelTrainingState(processes, name=name)

    def run(self, result_path: str):
        self.state.result_path = PathBuilder.build(f"{result_path}/{self.state.name}/")

        for process in self.state.processes:
            process.path = f"{self.state.result_path}{process.label}/"
            print(f'{datetime.datetime.now()}: MLModelTraining: starting training process for label {process.label}..\n')
            self.state.hp_items[process.label] = process.run(1)
            print(f'{datetime.datetime.now()}: MLModelTraining: finished training process for label {process.label}.\n')

        return self.state

    @staticmethod
    def get_documentation():
        doc = str(MLModelTrainingInstruction.__doc__)
        valid_metrics = str([metric.name.lower() for metric in Metric])[1:-1].replace("'", "`")

        mapping = {
            "metrics: a list of metrics that will be computed on the dataset (used for training), but which will not be used for fitting the model;":
                f"metrics: a list of metrics that will be computed on the dataset (used for training), but which will not be used for fitting the "
                f"model; valid values are {valid_metrics}",
            "optimization_metric: a metric that will be used for fitting the model": f"optimization_metric: a metric that will be used for fitting "
                                                                                     f"the model; valid values are {valid_metrics}"

        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
