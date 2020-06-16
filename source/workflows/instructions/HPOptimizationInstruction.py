from scripts.specification_util import update_docs_per_mapping
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.Metric import Metric
from source.hyperparameter_optimization.config.SplitConfig import SplitConfig
from source.hyperparameter_optimization.core.HPAssessment import HPAssessment
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.hyperparameter_optimization.strategy.HPOptimizationStrategy import HPOptimizationStrategy
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.Instruction import Instruction


class HPOptimizationInstruction(Instruction):
    """
    Class implementing hyper-parameter optimization and training and assessing the model through nested cross-validation (CV):

    The process is defined by two loops:
        - the outer loop over defined splits of the dataset for performance assessment
        - the inner loop over defined hyper-parameter space and with cross-validation or train & validation split
          to choose the best hyper-parameters.

    Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

    Arguments:

        dataset (Dataset): dataset to use for training and assessing the classifier

        hp_strategy (HPOptimizationStrategy): how to search different hyperparameters; common options include grid search, random search.
            Valid values are objects of any class inheriting :py:obj:`~source.hyperparameter_optimization.strategy.HPOptimizationStrategy.HPOptimizationStrategy`.

        hp_settings (list): a list of combinations of `preprocessing_sequence`, `encoding` and `ml_method`. `preprocessing_sequence` is optional,
            while `encoding` and `ml_method` are mandatory. These three options (and their parameters) can be optimized over, choosing the
            highest performing combination.

        assessment (SplitConfig): description of the outer loop (for assessment) of nested cross-validation. It describes how to split the
            data, how many splits to make, what percentage to use for training and what reports to execute on those splits. See SplitConfig.

        selection (SplitConfig): description of the inner loop (for selection) of nested cross-validation.
            The same as assessment argument, just to be executed in the inner loop. See SplitConfig.

        metrics (list): a list of metrics to compute for all splits and settings created during the nested cross-validation. These metrics
            will be computed only for reporting purposes. For choosing the optimal setting, `optimization_metric` will be used.

        optimization_metric (Metric): a metric to use for optimization and assessment in the nested cross-validation.

        label_configuration (LabelConfiguration): a list of labels for which to train the classifiers. The goal of the nested CV is to find
            the setting which will have best performance in predicting the given label
            (e.g. if a subject has experienced an immune event or not). Performance and optimal settings will be reported for each
            label separately.

        batch_size (int): how many processes should be created at once to speed up the analysis.
            For personal machines, 4 is usually a good choice.

        data_reports (list): a list of reports to be executed on the whole dataset.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_nested_cv_instruction: # user-defined name of the instruction
            type: HPOptimization # which instruction should be executed
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
                - celiac
            dataset: d1 # which dataset to use for the nested CV
            strategy: GridSearch # how to choose the combinations which to test from settings (GridSearch means test all)
            metrics: # list of metrics to compute for all settings, but these do not influence the choice of optimal model
                - accuracy
                - auc
            reports: # reports to execute on the dataset (before CV, splitting, encoding etc.)
                - rep1
            batch_size: 4 # number of parallel processes to create (could speed up the computation)
            optimization_metric: balanced_accuracy # the metric to use for choosing the optimal model and during training

    """

    def __init__(self, dataset, hp_strategy: HPOptimizationStrategy, hp_settings: list, assessment: SplitConfig, selection: SplitConfig,
                 metrics: set, optimization_metric: Metric, label_configuration: LabelConfiguration, path: str = None,
                 context: dict = None, batch_size: int = 1, data_reports: dict = None, name: str = None):
        self.hp_optimization_state = HPOptimizationState(dataset, hp_strategy, hp_settings, assessment, selection, metrics,
                                                         optimization_metric, label_configuration, path, context, batch_size, data_reports,
                                                         name)

    def run(self, result_path: str):
        self.hp_optimization_state.path = result_path
        state = HPAssessment.run_assessment(self.hp_optimization_state)
        self.print_performances(state)
        return state

    def print_performances(self, state: HPOptimizationState):
        print(f"Performances ({state.optimization_metric.name.lower()}) -----------------------------------------------")

        for label in state.label_configuration.get_labels_by_name():
            print(f"\n\nLabel: {label}")
            print(f"Performance ({state.optimization_metric.name.lower()}) per assessment split:")
            for split in range(state.assessment.split_count):
                print(f"Split {split+1}: {state.assessment_states[split].label_states[label].optimal_assessment_item.performance}")
            print(f"Average performance ({state.optimization_metric.name.lower()}): "
                  f"{sum([state.assessment_states[split].label_states[label].optimal_assessment_item.performance for split in range(state.assessment.split_count)])/state.assessment.split_count}")
            print("------------------------------")

    @staticmethod
    def get_documentation():
        doc = str(HPOptimizationInstruction.__doc__)
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
