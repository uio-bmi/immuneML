import warnings

import numpy as np
import yaml

from source.data_model.dataset.Dataset import Dataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class DesignMatrixExporter(EncodingReport):

    def __init__(self, dataset: Dataset = None, result_path: str = None):
        self.dataset = dataset
        self.result_path = result_path

    def generate(self):

        PathBuilder.build(self.result_path)

        self.export_matrix()
        self.export_details()
        self.export_labels()

    def export_matrix(self):
        if not isinstance(self.dataset.encoded_data.examples, np.ndarray):
            data = self.dataset.encoded_data.examples.toarray()
        else:
            data = self.dataset.encoded_data.examples
        np.savetxt(fname=f"{self.result_path}design_matrix.csv", X=data, delimiter=",",
                   header=",".join(self.dataset.encoded_data.feature_names), comments='')

    def export_details(self):
        with open(f"{self.result_path}encoding_details.yaml", "w") as file:
            details = {
                "feature_names": self.dataset.encoded_data.feature_names,
                "encoding": self.dataset.encoded_data.encoding,
                "example_ids": self.dataset.encoded_data.example_ids
            }

            yaml.dump(details, file)

    def export_labels(self):
        if self.dataset.encoded_data.labels is not None:
            labels = np.array([self.dataset.encoded_data.labels[l] for l in self.dataset.encoded_data.labels]).T
            np.savetxt(fname=f"{self.result_path}labels.csv", X=labels, delimiter=",",
                       header=",".join(self.dataset.encoded_data.labels.keys()), comments='')

    def check_prerequisites(self):
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn("DesignMatrixExporter: the dataset is not encoded, skipping this report...")
            return False
        else:
            return True

