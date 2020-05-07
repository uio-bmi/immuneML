import os
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from source.analysis.data_manipulation.DataReshaper import DataReshaper
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants


class TestDataReshaper(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    # 5 features, 5 repertoires. Each repertoire has 3 labels. Each feature has 2 annotations.
    encoded_data = {
        'examples': sparse.csr_matrix(np.array([
            [1, 2, 3, 4, 5],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [90, 10, 1, 3, 4],
            [0, 1, 1, 100, 200]
        ])),
        'example_ids': ["A", "B", "C", "D", "E"],
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

    dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data))

    def test_melt(self):
        result = DataReshaper.reshape(dataset=TestDataReshaper.dataset)

        self.assertTrue(result.shape[1] == 11)
        self.assertTrue(result.shape[0] == 25)
        self.assertTrue(result.isnull().any().sum() == 0)
