from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.Report import Report


class ExploratoryAnalysisUnit:

    def __init__(self, dataset: Dataset, report: Report, encoder: DatasetEncoder = None,
                 label_config: LabelConfiguration = None):
        self.dataset = dataset
        self.encoder = encoder
        self.report = report
        self.label_config = label_config
