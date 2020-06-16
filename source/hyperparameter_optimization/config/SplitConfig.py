from scripts.specification_util import update_docs_per_mapping
from source.hyperparameter_optimization.config.ReportConfig import ReportConfig
from source.hyperparameter_optimization.config.SplitType import SplitType


class SplitConfig:
    """
    SplitConfig describes how to split the data for cross-validation. It allows for the following combinations:

        - LOOCV (leave-one-out cross-validation)
        - K_FOLD (k-fold cross-validation)
        - RANDOM (Monte Carlo cross-validation - randomly splitting the dataset to training and test datasets)

    Arguments:

        split_strategy (SplitType): one of the three types of cross-validation listed above (`LOOCV`, `K_FOLD` or `RANDOM`)

        split_count (int): if split_strategy is `K_FOLD`, then this defined how many splits to make (K), if split_strategy is RANDOM,
            split_count defines how many random splits to make, resulting in split_count training/test dataset pairs, or if
            split_strategy is LOOCV, split_count does not need to be specified.

        training_percentage: if split_strategy is RANDOM, this defines which portion of the original dataset to use for creating
            the training dataset; for other values of split_strategy, this parameter is not used.

        reports (ReportConfig): defines which reports to execute on which datasets or settings. See ReportConfig for more details.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        # as a part of a HPOptimization instruction, defining the outer (assessment) loop of nested cross-validation:
        assessment: # outer loop of nested CV
            split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
            split_count: 5 # how many train/test datasets to generate
            training_percentage: 0.7 # what percentage of the original data should be used for the training set
            reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                data_splits: # list of reports to execute on training/test datasets (before they are encoded)
                    - rep1
                encoding: # list of reports to execute on encoded training/test datasets
                    - rep4
                hyperparameter: # list of reports to execute when nested CV is finished to show overall performance
                    - rep2

    """

    def __init__(self, split_strategy: SplitType, split_count: int, training_percentage: float = None, reports: ReportConfig = None):
        self.split_strategy = split_strategy
        self.split_count = split_count
        self.training_percentage = training_percentage
        self.reports = reports if reports is not None else ReportConfig()

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
            "reports (ReportConfig)": "reports"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
