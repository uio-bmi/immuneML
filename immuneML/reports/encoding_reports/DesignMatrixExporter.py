import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class DesignMatrixExporter(EncodingReport):
    """
    Exports the design matrix and related information of a given encoded Dataset to csv files. If the encoded data has more than 2 dimensions
    (such as when using the OneHot encoder with option Flatten=False), the data are instead exported to .npy format and can be imported later outside of
    immuneML using numpy package and numpy.load() function.

    There are no parameters for this report.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dme_report: DesignMatrixExporter

    """
    dataset: Dataset = None
    result_path: Path = None
    name: str = None

    @classmethod
    def build_object(cls, **kwargs):
        return DesignMatrixExporter(**kwargs)

    def _generate(self) -> ReportResult:

        PathBuilder.build(self.result_path)

        matrix_result = self._export_matrix()
        details_result = self._export_details()
        label_result = self._export_labels()

        return ReportResult(self.name, output_tables=[matrix_result], output_text=[details_result, label_result])

    def _export_matrix(self) -> ReportOutput:
        data = self._get_data()
        file_path = self._save_to_file(data, self.result_path / "design_matrix")
        return ReportOutput(file_path, "design matrix")

    def _get_data(self) -> np.ndarray:
        if not isinstance(self.dataset.encoded_data.examples, np.ndarray):
            data = self.dataset.encoded_data.examples.toarray()
        else:
            data = self.dataset.encoded_data.examples
        return data

    def _save_to_file(self, data: np.ndarray, file_path: Path) -> Path:
        if len(data.shape) <= 2:
            file_path = file_path.with_suffix(".csv")
            np.savetxt(fname=str(file_path), X=data, delimiter=",", comments='', header=",".join(self.dataset.encoded_data.feature_names))
        else:
            file_path = file_path.with_suffix(".npy")
            np.save(file_path, data)
        return file_path

    def _export_details(self) -> ReportOutput:
        file_path = self.result_path / "encoding_details.yaml"
        with file_path.open("w") as file:
            details = {
                "feature_names": self.dataset.encoded_data.feature_names,
                "encoding": self.dataset.encoded_data.encoding,
                "example_ids": list(self.dataset.encoded_data.example_ids)
            }

            yaml.dump(details, file)

        return ReportOutput(file_path, "encoding details")

    def _export_labels(self) -> ReportOutput:
        if self.dataset.encoded_data.labels is not None:
            labels_df = pd.DataFrame(self.dataset.encoded_data.labels)
            file_path = self.result_path / "labels.csv"
            labels_df.to_csv(file_path, sep=",", index=False)
            return ReportOutput(file_path, "exported labels")

    def check_prerequisites(self):
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn("DesignMatrixExporter: the dataset is not encoded, skipping this report...")
            return False
        else:
            return True

