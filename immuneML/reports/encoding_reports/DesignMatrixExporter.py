import warnings
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import os
import pandas as pd
import yaml
import zipfile

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class DesignMatrixExporter(EncodingReport):
    """
    Exports the design matrix and related information of a given encoded Dataset to csv files.
    If the encoded data has more than 2 dimensions (such as when using the OneHot encoder with option Flatten=False),
    the data are then exported to different formats to facilitate
    their import with external software.

    Args:

        format_file (str): the format and extension of the file to store the design matrix. The supported formats are:
        npy, csv, hdf5 and npy.zip, csv.zip, hdf5.zip.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dme_report: DesignMatrixExporter
            DesignMatrixExporter:
                format_file: .npy/.csv/.hdf5/.npy.zip/.cvs.zip/.hdf5.zip

    """

    def __init__(self, dataset = None, result_path = None,
                 name: str = None, format_file: str = 'csv'):
        super().__init__(name)
        self.format = format_file
        self.dataset = dataset
        self.result_path = result_path

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
        """Create a file for the design matrix in the desired format."""
        
        data = self._get_data()
        file_path = self.result_path / "design_matrix"

        # Check that the requested extension is supported.
        assert self.format.endswith(("hdf5", "hdf5.zip", "csv", "csv.zip", "npy", "npy.zip")), \
            f'Output format {self.format} not recognised for the Encoding Report'

        # Use h5py to create a hdf5 file.
        if self.format.endswith(("hdf5", "hdf5.zip")):
            file_path = file_path.with_suffix(".hdf5")
            with h5py.File(file_path, 'w') as hf_object:
                hf_object.create_dataset(str(file_path), data=data)
        # Use numpy to create a csv ord npy file.
        elif self.format.endswith(("csv", "csv.zip")):
            file_path = file_path.with_suffix(".csv")
            np.savetxt(fname=str(file_path), X=data, delimiter=",", comments='',
                       header=",".join(self.dataset.encoded_data.feature_names))
        else:
            file_path = file_path.with_suffix(".npy")
            np.save(str(file_path), data)
        
        # If requested, compress the file into a .zip.
        if self.format.endswith(".zip"):
            file_path_zip = str(file_path) + ".zip"
            with zipfile.ZipFile(file_path_zip, 'w') as zipped_file:
                zipped_file.write(file_path_zip, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(str(file_path))
        return ReportOutput(file_path, "design matrix")

    def _get_data(self) -> np.ndarray:
        if not isinstance(self.dataset.encoded_data.examples, np.ndarray):
            data = self.dataset.encoded_data.examples.toarray()
        else:
            data = self.dataset.encoded_data.examples
        return data

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
