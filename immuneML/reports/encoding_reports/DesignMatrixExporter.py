import logging
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class DesignMatrixExporter(EncodingReport):
    """
    Exports the design matrix and related information of a given encoded Dataset to csv files.
    If the encoded data has more than 2 dimensions (such as when using the OneHot encoder with option Flatten=False),
    the data are then exported to different formats to facilitate their import with external software.

    **Specification arguments:**

    - file_format (str): the format and extension of the file to store the design matrix. The supported formats are:
      npy, csv, hdf5, npy.zip, csv.zip or hdf5.zip.

    Note: when using hdf5 or hdf5.zip output formats, make sure the 'hdf5' dependency is installed.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_dme_report:
                    DesignMatrixExporter:
                        file_format: csv

    """
    def __init__(self, dataset: Dataset = None, result_path: Path = None, file_format: str = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.file_format = file_format

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()), ['file_format', 'name'], DesignMatrixExporter.__name__, DesignMatrixExporter.__name__)
        ParameterValidator.assert_in_valid_list(kwargs['file_format'], ['npy', 'csv', 'npy.zip', 'csv.zip', 'hdf5.zip'], DesignMatrixExporter.__name__, 'file_format')

        return DesignMatrixExporter(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        matrix_result = self._export_matrix()
        details_result = self._export_details()
        label_result = self._export_labels()

        return ReportResult(self.name,
                            info="The design matrix and related information of a given encoded Dataset",
                            output_tables=[matrix_result, label_result], output_text=[details_result])

    def _export_matrix(self) -> ReportOutput:
        """Create a file for the design matrix in the desired format."""
        
        data = self.dataset.encoded_data.get_examples_as_np_matrix()
        file_path = self.result_path / "design_matrix"
        ext = os.path.splitext(self.file_format)[0]
        file_path = file_path.with_suffix('.' + ext)

        # Use h5py to create a hdf5 file.
        if ext == "hdf5":
            import h5py
            with h5py.File(str(file_path), 'w') as hf_object:
                hf_object.create_dataset(str(file_path), data=data)

        # Use numpy to create a csv or npy file.
        elif len(data.shape) <= 2 and ext == "csv":
            feature_names = self.dataset.encoded_data.feature_names
            header = ",".join(str(name) for name in feature_names) if feature_names is not None else ""
            np.savetxt(fname=str(file_path), X=data, delimiter=",", comments='',
                       header=header)
        else:
            if ext != "npy":
                logging.info('The selected Report format is not compatible, .npy is used instead')
                file_path = file_path.with_suffix(".npy")
                ext = "npy"
            np.save(str(file_path), data)
        
        # If requested, compress the file into a .zip.
        if self.file_format.endswith(".zip"):
            file_path_zip = file_path.with_suffix('.' + ext + '.zip')
            with zipfile.ZipFile(str(file_path_zip), 'w') as zipped_file:
                zipped_file.write(str(file_path), compress_type=zipfile.ZIP_DEFLATED)
            os.remove(str(file_path)) 
            file_path = file_path_zip
        return ReportOutput(file_path, "design matrix")

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
