from unittest import TestCase
import shutil

import numpy as np
from scipy import sparse
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.encodings.pipeline.steps.FisherExactFeatureAnnotation import FisherExactFeatureAnnotation
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.BooleanType import BooleanType
from source.analysis.criteria_matches.OperationType import OperationType
from source.util.PathBuilder import PathBuilder


class TestFisherExactFeatureAnnotations(TestCase):

    # 5 features, 5 repertoires. Each repertoire has 3 labels. Each feature has 2 annotations.

    encoded_data = {
        'repertoires': sparse.csr_matrix(np.array([
            [1, 2, 3, 4, 5],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [90, 10, 1, 3, 4],
            [0, 1, 1, 100, 200]
        ])),
        'repertoire_ids': ["A", "B", "C", "D", "E"],
        'labels': {
            "diabetes": np.array(['T1D', 'CTL', 'FDR', 'CTL', 'T1D']),
            "aab": np.array([0, 0, 3, 1, 0])
        },
        'feature_names': ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
        'feature_annotations': pd.DataFrame({
            "feature": ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
            "sequence": ["AADAAA", "BBBBDB", "DDDDDE", "DDDDDD", "FFFFFF"],
            "v_gene": ["VGENE1", "VGENE2", "VGENE4", "VGENE6", "VGENE7"]
        })
    }

    dataset = Dataset(encoded_data=EncodedData(**encoded_data),
                      filenames=[filename + ".tsv" for filename in encoded_data["repertoire_ids"]])

    def test_transform(self):

        path = EnvironmentSettings.root_path + "test/tmp/fisherexactfeatureannotationsstep/"
        PathBuilder.build(path)

        step = FisherExactFeatureAnnotation(
            positive_criteria={
                "type": BooleanType.OR,
                "operands": [
                    {
                        "type": OperationType.IN,
                        "value": {
                            "type": DataType.COLUMN,
                            "name": "diabetes"
                        },
                        "allowed_values": ["T1D"]
                    },
                    {
                        "type": BooleanType.AND,
                        "operands": [
                            {
                                "type": OperationType.IN,
                                "value": {
                                    "type": DataType.COLUMN,
                                    "name": "diabetes"
                                },
                                "allowed_values": ["FDR"]
                            },
                            {
                                "type": OperationType.GREATER_THAN,
                                "value": {
                                    "type": DataType.COLUMN,
                                    "name": "aab"
                                },
                                "threshold": 2
                            }
                        ]
                    }
                ]
            },
            result_path=path,
            filename="encoded_dataset.pickle"
        )

        dataset = step.fit_transform(TestFisherExactFeatureAnnotations.dataset)
        self.assertTrue(round(dataset.encoded_data.feature_annotations["fisher_p.two_tail"][1], 1) == 0.4)

        shutil.rmtree(path)
