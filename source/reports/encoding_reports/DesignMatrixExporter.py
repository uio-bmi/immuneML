import warnings

import numpy as np
import pandas as pd
import yaml

from source.data_model.dataset.Dataset import Dataset
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class DesignMatrixExporter(EncodingReport):
    """
    Exports the design matrix and related information of a given encoded Dataset to csv files.


    Specification:

        definitions:
            datasets:
                my_dme_data:
                    ...
            encodings:
                my_dme_encoding:
                    ...
            reports:
                my_dme_report:
                    DesignMatrixExporter
        instructions:
                instruction_1:
                    type: ExploratoryAnalysis
                    analyses:
                        my_mr_analysis:
                            dataset: my_dme_data
                            encoding: my_dme_encoding
                            report: my_dme_report
                            labels:
                                - ...


    """

    @classmethod
    def build_object(cls, **kwargs):
        return DesignMatrixExporter(**kwargs)

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
            labels_df = pd.DataFrame(self.dataset.encoded_data.labels)
            labels_df.to_csv(f"{self.result_path}labels.csv", sep=",", index=False)

    def check_prerequisites(self):
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
            warnings.warn("DesignMatrixExporter: the dataset is not encoded, skipping this report...")
            return False
        else:
            return True

