class ReportConfig:

    def __init__(self, data_splits: list = None, models: list = None, optimal_models: list = None, performance: list = None,
                 data: list = None):

        self.data_split_reports = data_splits if data_splits is not None else []
        self.model_reports = models if models is not None else []
        self.optimal_model_reports = optimal_models if optimal_models is not None else []
        self.performance_reports = performance if performance is not None else []
        self.data_reports = data if data is not None else []
