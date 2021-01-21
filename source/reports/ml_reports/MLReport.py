from source.data_model.dataset.Dataset import Dataset
from source.hyperparameter_optimization import HPSetting
from source.ml_methods.MLMethod import MLMethod
from source.reports.Report import Report


class MLReport(Report):
    """
    ML model reports show some type of features or statistics about one trained ML model.

    In the :ref:`TrainMLModel` instruction, ML model reports can be specified inside the 'selection' or 'assessment' specification under the key 'reports/models'.
    """

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: str = None, name: str = None, hp_setting: HPSetting = None, label=None):
        super().__init__(name)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.method = method
        self.result_path = result_path
        self.name = name
        self.hp_setting = hp_setting
        self.label = label

    @staticmethod
    def get_title():
        return "ML model reports"
