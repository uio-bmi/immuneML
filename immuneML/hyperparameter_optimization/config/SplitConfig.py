from immuneML.hyperparameter_optimization.config.LeaveOneOutConfig import LeaveOneOutConfig
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.ReportConfig import ReportConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from scripts.specification_util import update_docs_per_mapping


class SplitConfig:
    """
    SplitConfig describes how to split the data for cross-validation. It allows for the following combinations:

        - loocv (leave-one-out cross-validation)

        - k_fold (k-fold cross-validation)

        - random (Monte Carlo cross-validation - randomly splitting the dataset to training and test datasets)

        - manual (train and test dataset are explicitly specified by providing metadata files for the two datasets - currently available only for repertoire datasets)

        - leave_one_out_stratification (leave-one-out CV where one refers to a specific parameter, e.g. if subject is known in a receptor dataset, it is possible to have leave-subject-out CV - currently only available for receptor datasets).

    Arguments:

        split_strategy (SplitType): one of the three types of cross-validation listed above (`LOOCV`, `K_FOLD` or `RANDOM`)

        split_count (int): if split_strategy is `K_FOLD`, then this defined how many splits to make (K), if split_strategy is RANDOM, split_count defines how many random splits to make, resulting in split_count training/test dataset pairs, or if split_strategy is `LOOCV`, `MANUAL` or `LEAVE_ONE_OUT_STRATIFICATION`, split_count does not need to be specified.

        training_percentage: if split_strategy is RANDOM, this defines which portion of the original dataset to use for creating the training dataset; for other values of split_strategy, this parameter is not used.

        reports (ReportConfig): defines which reports to execute on which datasets or settings. See :ref:`ReportConfig` for more details.

        manual_config (:py:obj:`~immuneML.hyperparameter_optimization.config.ManualSplitConfig.ManualSplitConfig`): if split strategy is `MANUAL`,
        here the paths to metadata files should be given (fields `train_metadata_path` and `test_metadata_path`). The matching of examples is done
        using the "subject_id" field so it has to be present in both the original dataset and the metadata files provided here. Manual splitting to
        train and test dataset is currently supported only for repertoire datasets. If split strategy is anything else, this field has no effect
        and can be omitted.

        leave_one_out_config (:py:obj:`~immuneML.hyperparameter_optimization.config.LeaveOneOutConfig.LeaveOneOutConfig`): if split strategy is
        `LEAVE_ONE_OUT_STRATIFICATION`, this config describes which parameter to use for stratification thus making a list of train/test dataset
        combinations in which in the test set there are examples with only one value of the specified parameter. `leave_one_out_config` argument
        accepts two inputs: `parameter` which is the name of the parameter to use for stratification and `min_count` which defines the minimum
        number of examples that can be present in the test dataset. This type of generating train and test datasets is only supported for receptor
        datasets so far. If split strategy is anything else, this field has no effect and can be omitted.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        # as a part of a TrainMLModel instruction, defining the outer (assessment) loop of nested cross-validation:
        assessment: # outer loop of nested CV
            split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
            split_count: 5 # how many train/test datasets to generate
            training_percentage: 0.7 # what percentage of the original data should be used for the training set
            reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                data_splits: # list of data reports to execute on training/test datasets (before they are encoded)
                    - rep1
                encoding: # list of encoding reports to execute on encoded training/test datasets
                    - rep2
                models: # list of ML model reports to execute on the trained classifiers in the assessment loop
                    - rep3

        # as a part of a TrainMLModel instruction, defining the inner (selection) loop of nested cross-validation:
        selection: # inner loop of nested CV
            split_strategy: leave_one_out_stratification
            leave_one_out_config: # perform leave-(subject)-out CV
                parameter: subject # which parameter to use for splitting, must be present in the metadata for each example
                min_count: 1 # what is the minimum number of examples with unique value of the parameter specified above for the analysis to be valid
            reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                data_splits: # list of data reports to execute on training/test datasets (before they are encoded)
                    - rep1
                encoding: # list of encoding reports to execute on encoded training/test datasets
                    - rep2
                encoding: # list of ML model reports to execute the trained classifiers in the selection loop
                    - rep3

    """

    def __init__(self, split_strategy: SplitType, split_count: int, training_percentage: float = None, reports: ReportConfig = None,
                 manual_config: ManualSplitConfig = None, leave_one_out_config: LeaveOneOutConfig = None):
        self.split_strategy = split_strategy
        self.split_count = split_count
        self.training_percentage = training_percentage
        self.reports = reports if reports is not None else ReportConfig()
        self.manual_config = manual_config
        self.leave_one_out_config = leave_one_out_config

    def __str__(self):
        desc = ""
        if self.split_strategy == SplitType.K_FOLD:
            desc = f"{self.split_count}-fold CV"
        elif self.split_strategy == SplitType.RANDOM:
            desc = f"{self.split_count}-fold MC CV (training percentage: {self.training_percentage})"
        elif self.split_strategy == SplitType.LOOCV:
            desc = "LOOCV"
        return desc

    @staticmethod
    def get_documentation():
        doc = str(SplitConfig.__doc__)
        mapping = {
            "split_strategy (SplitType)": "split_strategy",
            "reports (ReportConfig)": "reports",
            "manual_config (:py:obj:`~immuneML.hyperparameter_optimization.config.ManualSplitConfig.ManualSplitConfig`)": "manual_config",
            "leave_one_out_config (:py:obj:`~immuneML.hyperparameter_optimization.config.LeaveOneOutConfig.LeaveOneOutConfig`)": "leave_one_out_config"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
