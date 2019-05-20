import abc

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.reports.Report import Report


class EncodingReport(Report):

    def generate_report(self, params):
        assert isinstance(params["dataset"].encoded_data, EncodedData), \
            "EncodingReport: it is necessary that the passed dataset is encoded before the report is generated. Encode the" \
            "dataset and try again."
        return self.generate(dataset=params["dataset"], result_path=params["result_path"], params=params)

    @abc.abstractmethod
    def generate(self, dataset: Dataset, result_path: str, params: dict):
        pass
