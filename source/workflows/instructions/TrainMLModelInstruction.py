import datetime
from collections import Counter

from scripts.specification_util import update_docs_per_mapping
from source.IO.ml_method.MLExporter import MLExporter
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.core.HPAssessment import HPAssessment
from source.hyperparameter_optimization.core.HPUtil import HPUtil
from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.MLProcess import MLProcess


class TrainMLModelInstruction(Instruction):
    """
    Class implementing hyperparameter optimization and training and assessing the model through nested cross-validation (CV).
    The process is defined by two loops:

        - the outer loop over defined splits of the dataset for performance assessment

        - the inner loop over defined hyperparameter space and with cross-validation or train & validation split
          to choose the best hyperparameters.

    Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

    Arguments:

        dataset (Dataset): dataset to use for training and assessing the classifier

        hp_strategy (HPOptimizationStrategy): how to search different hyperparameters; common options include grid search, random search. Valid values are objects of any class inheriting :py:obj:`~source.hyperparameter_optimization.strategy.HPOptimizationStrategy.HPOptimizationStrategy`.

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

        batch_size (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

        data_reports (list): a list of reports to be executed on the whole dataset.

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
                        - rep4
                    hyperparameter: # list of reports to execute when nested CV is finished to show overall performance
                        - rep2
            selection: # inner loop of nested CV
                split_strategy: k_fold # perform k-fold CV
                split_count: 5 # how many fold to create: here these two parameters mean: do 5-fold CV
                reports:
                    data_splits: # list of reports to execute on training/test datasets (in the inner loop, so these are actually training and validation datasets)
                        - rep1
                    models: # list of reports to execute on trained ML methods
                        - rep3
                    optimal_models: [] # list of reports to execute on optimal ML methods for each CV split
                    encoding: # list of reports to execute on encoded training/test datasets (again, it is training/validation here)
                        - rep4
            labels: # list of labels to optimize the classifier for, as given in the metadata for the dataset
                - celiac:
                    positive_class: '+' # if it's binary classification, positive class parameter should be set
                - T1D # this is not binary label, so no need to specify positive class
            dataset: d1 # which dataset to use for the nested CV
            strategy: GridSearch # how to choose the combinations which to test from settings (GridSearch means test all)
            metrics: # list of metrics to compute for all settings, but these do not influence the choice of optimal model
                - accuracy
                - auc
            reports: # reports to execute on the dataset (before CV, splitting, encoding etc.)
                - rep1
            batch_size: 4 # number of parallel processes to create (could speed up the computation)
            optimization_metric: balanced_accuracy # the metric to use for choosing the optimal model and during training
            refit_optimal_model: False # use trained model, do not refit on the full dataset
            store_encoded_data: True # store encoded datasets in pickle format

    """

    def __init__(self, dataset, hp_strategy: HPOptimizationStrategy, hp_settings: list, assessment: SplitConfig, selection: SplitConfig,
                 metrics: set, optimization_metric: Metric, label_configuration: LabelConfiguration, path: str = None, context: dict = None,
                 batch_size: int = 1, data_reports: dict = None, name: str = None, refit_optimal_model: bool = False, store_encoded_data: bool = None):
        self.state = TrainMLModelState(dataset, hp_strategy, hp_settings, assessment, selection, metrics,
                                       optimization_metric, label_configuration, path, context, batch_size,
                                       data_reports if data_reports is not None else {}, name, refit_optimal_model, store_encoded_data)

    def run(self, result_path: str):
        self.state.path = result_path
        self.state = HPAssessment.run_assessment(self.state)
        self._compute_optimal_hp_item_per_label()
        self.state.hp_report_results = HPUtil.run_hyperparameter_reports(self.state, f"{self.state.path}hyperparameter_reports/")
        self.print_performances(self.state)
        return self.state

    def _compute_optimal_hp_item_per_label(self):
        n_labels = self.state.label_configuration.get_label_count()

        for idx, label in enumerate(self.state.label_configuration.get_labels_by_name()):
            self._compute_optimal_item(label, f"(label {idx + 1} / {n_labels})")
            zip_path = MLExporter.export_zip(hp_item=self.state.optimal_hp_items[label], path=f"{self.state.path}optimal_{label}/", label=label)
            self.state.optimal_hp_item_paths[label] = zip_path

    def _compute_optimal_item(self, label: str, index_repr: str):
        optimal_hp_settings = [state.label_states[label].optimal_hp_setting for state in self.state.assessment_states]
        optimal_hp_setting = Counter(optimal_hp_settings).most_common(1)[0][0]
        if self.state.refit_optimal_model:
            print(f"{datetime.datetime.now()}: Hyperparameter optimization: retraining optimal model for label {label} {index_repr}.\n", flush=True)
            self.state.optimal_hp_items[label] = MLProcess(self.state.dataset, None, label, self.state.metrics, self.state.optimization_metric,
                                                           f"{self.state.path}optimal_{label}/", number_of_processes=self.state.batch_size,
                                                           label_config=self.state.label_configuration, hp_setting=optimal_hp_setting,
                                                           store_encoded_data=self.state.store_encoded_data).run(0)
            print(f"{datetime.datetime.now()}: Hyperparameter optimization: finished retraining optimal model for label {label} {index_repr}.\n", flush=True)

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
            "Valid values are objects of any class inheriting :py:obj:`~source.hyperparameter_optimization.strategy."
            "HPOptimizationStrategy.HPOptimizationStrategy`.": f"Valid values are: {valid_strategies}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
