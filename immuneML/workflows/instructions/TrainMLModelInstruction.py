import datetime
from collections import Counter
from pathlib import Path

import pandas as pd

from immuneML.IO.ml_method.MLExporter import MLExporter
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Metric import Metric
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.core.HPAssessment import HPAssessment
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.MLProcess import MLProcess
from scripts.specification_util import update_docs_per_mapping


class TrainMLModelInstruction(Instruction):
    """
    Class implementing hyperparameter optimization and training and assessing the model through nested cross-validation (CV).
    The process is defined by two loops:

        - the outer loop over defined splits of the dataset for performance assessment

        - the inner loop over defined hyperparameter space and with cross-validation or train & validation split
          to choose the best hyperparameters.

    Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

    Note: If you are interested in plotting the performance of all combinations of encodings and ML methods on the test set,
    consider running the :ref:`MLSettingsPerformance` report as hyperparameter report in the assessment loop.


    Arguments:

        dataset (Dataset): dataset to use for training and assessing the classifier

        hp_strategy (HPOptimizationStrategy): how to search different hyperparameters; common options include grid search, random search. Valid values are objects of any class inheriting :py:obj:`~immuneML.hyperparameter_optimization.strategy.HPOptimizationStrategy.HPOptimizationStrategy`.

        hp_settings (list): a list of combinations of `preprocessing_sequence`, `encoding` and `ml_method`. `preprocessing_sequence` is optional, while `encoding` and `ml_method` are mandatory. These three options (and their parameters) can be optimized over, choosing the highest performing combination.

        assessment (SplitConfig): description of the outer loop (for assessment) of nested cross-validation. It describes how to split the data, how many splits to make, what percentage to use for training and what reports to execute on those splits. See :ref:`SplitConfig`.

        selection (SplitConfig): description of the inner loop (for selection) of nested cross-validation. The same as assessment argument, just to be executed in the inner loop. See :ref:`SplitConfig`.

        metrics (list): a list of metrics to compute for all splits and settings created during the nested cross-validation. These metrics will be computed only for reporting purposes. For choosing the optimal setting, `optimization_metric` will be used.

        optimization_metric (Metric): a metric to use for optimization and assessment in the nested cross-validation.

        label_configuration (LabelConfiguration): a list of labels for which to train the classifiers. The goal of the nested CV is to find the
        setting which will have best performance in predicting the given label (e.g., if a subject has experienced an immune event or not).
        Performance and optimal settings will be reported for each label separately. If a label is binary, instead of specifying only its name, one
        should explicitly set the name of the positive class as well under parameter `positive_class`. If positive class is not set, one of the label
        classes will be assumed to be positive.

        number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

        reports (list): a list of report names to be executed after the nested CV has finished to show the overall performance or some statistic;
        the reports to be specified here have to be :py:obj:`immuneML.reports.train_ml_model_reports.TrainMLModelReport.TrainMLModelReport` reports.

        refit_optimal_model (bool): if the final combination of preprocessing-encoding-ML model should be refitted on the full dataset thus providing
        the final model to be exported from instruction; alternatively, train combination from one of the assessment folds will be used

        store_encoded_data (bool): if the encoded datasets should be stored, can be True or False; setting this argument to True might increase the
        disk usage significantly

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_nested_cv_instruction: # user-defined name of the instruction
            type: TrainMLModel # which instruction should be executed
            settings: # a list of combinations of preprocessing, encoding and ml_method to optimize over
                - preprocessing: seq1 # preprocessing is optional
                  encoding: e1 # mandatory field
                  ml_method: simpleLR # mandatory field
                - preprocessing: seq1 # the second combination
                  encoding: e2
                  ml_method: simpleLR
            assessment: # outer loop of nested CV
                split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
                split_count: 1 # how many train/test datasets to generate
                training_percentage: 0.7 # what percentage of the original data should be used for the training set
                reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                    data_splits: # list of reports to execute on training/test datasets (before they are encoded)
                        - rep1
                    encoding: # list of reports to execute on encoded training/test datasets
                        - rep2
                    models: # list of reports to execute on trained ML methods for each assessment CV split
                        - rep3
            selection: # inner loop of nested CV
                split_strategy: k_fold # perform k-fold CV
                split_count: 5 # how many fold to create: here these two parameters mean: do 5-fold CV
                reports:
                    data_splits: # list of reports to execute on training/test datasets (in the inner loop, so these are actually training and validation datasets)
                        - rep1
                    models: # list of reports to execute on trained ML methods for each selection CV split
                        - rep2
                    encoding: # list of reports to execute on encoded training/test datasets (again, it is training/validation here)
                        - rep3
            labels: # list of labels to optimize the classifier for, as given in the metadata for the dataset
                - celiac:
                    positive_class: + # if it's binary classification, positive class parameter should be set
                - T1D # this is not binary label, so no need to specify positive class
            dataset: d1 # which dataset to use for the nested CV
            strategy: GridSearch # how to choose the combinations which to test from settings (GridSearch means test all)
            metrics: # list of metrics to compute for all settings, but these do not influence the choice of optimal model
                - accuracy
                - auc
            reports: # list of reports to execute when nested CV is finished to show overall performance
                - rep4
            number_of_processes: 4 # number of parallel processes to create (could speed up the computation)
            optimization_metric: balanced_accuracy # the metric to use for choosing the optimal model and during training
            refit_optimal_model: False # use trained model, do not refit on the full dataset
            store_encoded_data: True # store encoded datasets in pickle format

    """

    def __init__(self, dataset, hp_strategy: HPOptimizationStrategy, hp_settings: list, assessment: SplitConfig, selection: SplitConfig,
                 metrics: set, optimization_metric: Metric, label_configuration: LabelConfiguration, path: Path = None, context: dict = None,
                 number_of_processes: int = 1, reports: dict = None, name: str = None, refit_optimal_model: bool = False, store_encoded_data: bool = None):
        self.state = TrainMLModelState(dataset, hp_strategy, hp_settings, assessment, selection, metrics,
                                       optimization_metric, label_configuration, path, context, number_of_processes,
                                       reports if reports is not None else {}, name, refit_optimal_model, store_encoded_data)

    def run(self, result_path: Path):
        self.state.path = result_path
        self.state = HPAssessment.run_assessment(self.state)
        self._compute_optimal_hp_item_per_label()
        self.state.report_results = HPUtil.run_hyperparameter_reports(self.state, self.state.path / "reports")
        self.print_performances(self.state)
        self._export_all_performances_to_csv()
        return self.state

    def _compute_optimal_hp_item_per_label(self):
        n_labels = self.state.label_configuration.get_label_count()

        for idx, label in enumerate(self.state.label_configuration.get_labels_by_name()):
            self._compute_optimal_item(label, f"(label {idx + 1} / {n_labels})")
            zip_path = MLExporter.export_zip(hp_item=self.state.optimal_hp_items[label], path=self.state.path / f"optimal_{label}", label=label)
            self.state.optimal_hp_item_paths[label] = zip_path

    def _compute_optimal_item(self, label: str, index_repr: str):
        optimal_hp_settings = [state.label_states[label].optimal_hp_setting for state in self.state.assessment_states]
        optimal_hp_setting = Counter(optimal_hp_settings).most_common(1)[0][0]
        if self.state.refit_optimal_model:
            print(f"{datetime.datetime.now()}: TrainMLModel: retraining optimal model for label {label} {index_repr}.\n", flush=True)
            self.state.optimal_hp_items[label] = MLProcess(self.state.dataset, None, label, self.state.metrics, self.state.optimization_metric,
                                                           self.state.path / f"optimal_{label}", number_of_processes=self.state.number_of_processes,
                                                           label_config=self.state.label_configuration, hp_setting=optimal_hp_setting,
                                                           store_encoded_data=self.state.store_encoded_data).run(0)
            print(f"{datetime.datetime.now()}: TrainMLModel: finished retraining optimal model for label {label} {index_repr}.\n", flush=True)

        else:
            optimal_assessment_state = self.state.assessment_states[optimal_hp_settings.index(optimal_hp_setting)]
            self.state.optimal_hp_items[label] = optimal_assessment_state.label_states[label].optimal_assessment_item

    def print_performances(self, state: TrainMLModelState):
        print(f"Performances ({state.optimization_metric.name.lower()}) -----------------------------------------------", flush=True)

        for label in state.label_configuration.get_labels_by_name():
            print(f"\n\nLabel: {label}", flush=True)
            print(f"Performance ({state.optimization_metric.name.lower()}) per assessment split:", flush=True)
            for split in range(state.assessment.split_count):
                print(f"Split {split+1}: {state.assessment_states[split].label_states[label].optimal_assessment_item.performance[state.optimization_metric.name.lower()]}", flush=True)
            print(f"Average performance ({state.optimization_metric.name.lower()}): "
                  f"{sum([state.assessment_states[split].label_states[label].optimal_assessment_item.performance[state.optimization_metric.name.lower()] for split in range(state.assessment.split_count)])/state.assessment.split_count}", flush=True)
            print("------------------------------", flush=True)

    def _export_all_performances_to_csv(self):
        self._export_optimal_performances_to_csv()
        self._export_assessment_performances_to_csv()
        if not (self.state.selection.training_percentage == 1 and self.state.selection.split_strategy == SplitType.RANDOM):
            self._export_selection_performance_to_csv()

    def _export_optimal_performances_to_csv(self):
        for label in self.state.label_configuration.get_labels_by_name():
            performance = {"hp_setting": [], "split": [], **{metric.name.lower(): [] for metric in self.state.metrics}}
            for index, assessment_state in enumerate(self.state.assessment_states):
                performance['split'].append(index+1)
                performance['hp_setting'].append(assessment_state.label_states[label].optimal_hp_setting.get_key())
                for metric in self.state.metrics:
                    performance[metric.name.lower()].append(assessment_state.label_states[label].optimal_assessment_item.performance[metric.name.lower()])
            pd.DataFrame(performance).to_csv(self.state.path / f"{label}_optimal_models_performance.csv", index=False)

    def _export_assessment_performances_to_csv(self):
        for label in self.state.label_configuration.get_labels_by_name():
            performance = {'hp_setting': [], 'split': [], 'optimal': [], **{metric.name.lower(): [] for metric in self.state.metrics}}
            for index, assessment_state in enumerate(self.state.assessment_states):
                for hp_setting, hp_item in assessment_state.label_states[label].assessment_items.items():
                    performance['hp_setting'].append(str(hp_setting))
                    performance['split'].append(index+1)
                    performance['optimal'].append(hp_setting == assessment_state.label_states[label].optimal_hp_setting)
                    for metric in self.state.metrics:
                        performance[metric.name.lower()].append(hp_item.performance[metric.name.lower()])
            pd.DataFrame(performance).to_csv(self.state.path / f"{label}_all_assessment_performances.csv", index=False)

    def _export_selection_performance_to_csv(self):
        for label in self.state.label_configuration.get_labels_by_name():
            for index, assessment_state in enumerate(self.state.assessment_states):
                selection_state = assessment_state.label_states[label].selection_state
                performance = {'hp_setting': [], 'split': [], **{metric.name.lower(): [] for metric in self.state.metrics}}
                for hp_setting, hp_item_list in selection_state.hp_items.items():
                    for i, hp_item in enumerate(hp_item_list):
                        performance['hp_setting'].append(hp_setting)
                        performance['split'].append(i+1)
                        for metric in self.state.metrics:
                            performance[metric.name.lower()].append(hp_item.performance[metric.name.lower()])
                pd.DataFrame(performance).to_csv(self.state.path / f"{label}_assessment_split_{index+1}_selection_performance.csv", index=False)

    @staticmethod
    def get_documentation():
        doc = str(TrainMLModelInstruction.__doc__)
        valid_values = str([metric.name.lower() for metric in Metric])[1:-1].replace("'", "`")
        valid_strategies = str(ReflectionHandler.all_nonabstract_subclass_basic_names(HPOptimizationStrategy, "",
                                                                                      "hyperparameter_optimization/strategy/"))[1:-1]\
            .replace("'", "`")
        mapping = {
            "dataset (Dataset)": "dataset",
            "hp_strategy (HPOptimizationStrategy)": "strategy",
            "hp_settings": "settings",
            "assessment (SplitConfig)": "assessment",
            "selection (SplitConfig)": "selection",
            "optimization_metric (Metric)": "optimization_metric",
            "label_configuration (LabelConfiguration)": "labels (list)",
            "data_reports": "reports",
            "a list of metrics": f"a list of metrics ({valid_values})",
            "a metric to use for optimization": f"a metric to use for optimization (one of {valid_values})",
            "Valid values are objects of any class inheriting :py:obj:`~immuneML.hyperparameter_optimization.strategy."
            "HPOptimizationStrategy.HPOptimizationStrategy`.": f"Valid values are: {valid_strategies}.",
            "the reports to be specified here have to be :py:obj:`immuneML.reports.train_ml_model_reports.TrainMLModelReport.TrainMLModelReport` reports.": f"the reports that can be provided here are :ref:`{TrainMLModelReport.get_title()}`."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
