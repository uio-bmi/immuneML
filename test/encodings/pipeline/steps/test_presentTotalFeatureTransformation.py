import random
import string
from unittest import TestCase
import shutil

import numpy as np
from scipy import sparse
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.encodings.pipeline.steps.PresentTotalFeatureTransformation import PresentTotalFeatureTransformation
from source.util.PathBuilder import PathBuilder
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.analysis.criteria_matches.BooleanType import BooleanType


class TestPresentTotalFeatureTransformation(TestCase):

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
            "diabetes": ['T1D', 'CTL', 'FDR', 'CTL', 'T1D'],
        },
        'feature_names': ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
        'feature_annotations': pd.DataFrame({
            "feature": ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
            "sequence": ["AADAAA", "BBBBDB", "DDDDDE", "DDDDDD", "FFFFFF"],
            "v_gene": ["VGENE1", "VGENE2", "VGENE4", "VGENE6", "VGENE7"],
            "matching_specificity": ["flu", "ebv", "GAD", "PPI", "GAD"],
            "p_val": [0.01, 0.00001, 0.000001, 0.01, 0.01],
            "odds_ratio": [0.51, 0, 0, 0, 0],
            "a": ["yes", "no", "no", "no", "no"],
            "b": ["no", "yes", "no", "no", "no"]
        })
    }

    dataset = Dataset(encoded_data=EncodedData(**encoded_data),
                      filenames=[filename + ".tsv" for filename in encoded_data["repertoire_ids"]])

    def test_transform(self):
        path = EnvironmentSettings.root_path + "test/tmp/presenttotalfeaturetransformation/"
        PathBuilder.build(path)

        filter_params = {
            "type": OperationType.IN,
            "allowed_values": ["GAD", "PPI"],
            "value": {
                "type": DataType.COLUMN,
                "name": "matching_specificity"
            }
        }

        step = PresentTotalFeatureTransformation(
            criteria=filter_params,
            result_path=path,
            filename="encoded_dataset.pickle")

        dataset = step.fit_transform(TestPresentTotalFeatureTransformation.dataset)

        self.assertTrue(dataset.encoded_data.repertoires.A[0, 0] == 3)
        self.assertTrue(dataset.encoded_data.repertoires.A[0, 1] == 5)

        shutil.rmtree(path)
