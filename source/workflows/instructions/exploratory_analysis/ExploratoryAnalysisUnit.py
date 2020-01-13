from source.data_model.dataset.Dataset import Dataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.Report import Report


class ExploratoryAnalysisUnit:

    def __init__(self, dataset: Dataset, report: Report, preprocessing_sequence: list = None, encoder: DatasetEncoder = None,
                 label_config: LabelConfiguration = None, batch_size: int = 1):
        self.dataset = dataset
        self.preprocessing_sequence = preprocessing_sequence
        self.encoder = encoder
        self.report = report
        self.label_config = label_config
        self.batch_size = batch_size
