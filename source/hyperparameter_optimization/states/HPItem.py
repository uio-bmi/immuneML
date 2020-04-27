class HPItem:

    def __init__(self, method=None, performance=None, hp_setting=None, train_predictions_path=None, test_predictions_path=None,
                 ml_details_path=None, train_dataset=None, test_dataset=None, split_index=None):
        self.method = method
        self.performance = performance
        self.hp_setting = hp_setting
        self.train_predictions_path = train_predictions_path
        self.test_predictions_path = test_predictions_path
        self.ml_details_path = ml_details_path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split_index = split_index
        self.model_report_results = []
