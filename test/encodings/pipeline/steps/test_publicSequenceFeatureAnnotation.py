import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.pipeline.steps.PublicSequenceFeatureAnnotation import PublicSequenceFeatureAnnotation
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestPublicSequenceFeatureAnnotation(TestCase):

    # 5 features, 5 repertoires. Each repertoire has 3 labels. Each feature has 2 annotations.
    encoded_data = {
        'examples': sparse.csr_matrix(np.array([
            [1, 2, 3, 4, 5],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [90, 10, 1, 3, 0],
            [0, 1, 1, 100, 0]
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

    dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data),
                                repertoires=[Repertoire(EnvironmentSettings.root_path + "test/tmp/publicsequencefeatureannotation/0.npy",
                                                        "", identifier) for identifier in encoded_data["example_ids"]])

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_transform(self):
        path = EnvironmentSettings.root_path + "test/tmp/publicsequencefeatureannotation/"
        PathBuilder.build(path)

        step = PublicSequenceFeatureAnnotation(
            result_path=path,
            filename="encoded_dataset.pickle",
        )
        dataset = step.fit_transform(TestPublicSequenceFeatureAnnotation.dataset)

        self.assertTrue("public_number_of_repertoires" in dataset.encoded_data.feature_annotations.columns)
        self.assertTrue(dataset.encoded_data.feature_annotations["public_number_of_repertoires"][4] == 1)

        shutil.rmtree(path)
