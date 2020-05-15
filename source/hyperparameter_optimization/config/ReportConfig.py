class ReportConfig:

    def __init__(self, data_splits: dict = None, models: dict = None, optimal_models: dict = None,
                 data: dict = None, encoding: dict = None, hyperparameter: dict = None):

        self.data_split_reports = data_splits if data_splits is not None else {}
        self.encoding_reports = encoding if encoding is not None else {}
        self.model_reports = models if models is not None else {}
        self.optimal_model_reports = optimal_models if optimal_models is not None else {}
        self.data_reports = data if data is not None else {}
        self.hyperparameter_reports = hyperparameter if hyperparameter is not None else {}
