import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from source.analysis.AxisType import AxisType
from source.analysis.data_manipulation.GroupSummarizationType import GroupSummarizationType
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.encodings.pipeline.steps.GroupDataTransformation import GroupDataTransformation
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestGroupByAnnotationSummation(TestCase):

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
            "diabetes": ['diabetes pos', 'diabetes neg', 'diabetes neg', 'diabetes pos', 'diabetes pos'],
            "celiac": ['celiac pos', 'celiac pos', 'celiac pos', 'celiac neg', 'celiac pos'],
            "cmv": ['cmv pos', 'cmv neg', 'cmv pos', 'cmv pos', 'cmv neg']
        },
        'feature_names': ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
        'feature_annotations': pd.DataFrame({
            "feature": ["VGENE1///AADAAA", "VGENE2///BBBBDB", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
            "sequence": ["AADAAA", "BBBBDB", "DDDDDE", "DDDDDD", "FFFFFF"],
            "v_gene": ["VGENE1", "VGENE2", "VGENE4", "VGENE6", "VGENE7"],
            "specificity": ["cmv", "ebv", "cmv", "gluten", "gluten"],
            "something": ["a", "b", "b", "a", "a"],
            "p_val": [0.01, 0.00001, 0.1, 0, 0.0000001]
        })
    }

    dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data))
    dataset.repertoire_ids = ["1", "2", "3", "4", "5"]

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_group_repertoires_1(self):

        path = EnvironmentSettings.root_path + "test/tmp/groupbyannotationsummationstep/"

        PathBuilder.build(path)

        group_columns = ["diabetes"]

        step = GroupDataTransformation(axis=AxisType.REPERTOIRES,
                                       group_summarization_type=GroupSummarizationType.AVERAGE,
                                       group_columns=group_columns,
                                       result_path=path)

        dataset = step.fit_transform(TestGroupByAnnotationSummation.dataset)

        encoded = dataset.encoded_data

        self.assertTrue(pd.DataFrame(encoded.labels).shape[0] == encoded.examples.shape[0])
        self.assertTrue(len(encoded.example_ids) == encoded.examples.shape[0])
        self.assertTrue(encoded.examples.shape[0] == 2)
        self.assertTrue(encoded.examples.shape[1] == 5)
        self.assertTrue(np.equal(encoded.examples[0, 0], 0.5))
        self.assertTrue(np.equal(encoded.examples[1, 0], 30.333333333333332))

        shutil.rmtree(path)

    def test_group_repertoires_2(self):

        path = EnvironmentSettings.root_path + "test/tmp/groupbyannotationsummationstep/"

        PathBuilder.build(path)

        group_columns = ["diabetes", "celiac"]

        step = GroupDataTransformation(axis=AxisType.REPERTOIRES,
                                       group_summarization_type=GroupSummarizationType.AVERAGE,
                                       group_columns=group_columns,
                                       result_path=path)

        dataset = step.fit_transform(TestGroupByAnnotationSummation.dataset)

        encoded = dataset.encoded_data

        self.assertTrue(pd.DataFrame(encoded.labels).shape[0] == encoded.examples.shape[0])
        self.assertTrue(len(encoded.example_ids) == encoded.examples.shape[0])
        self.assertTrue(encoded.examples.shape[0] == 3)
        self.assertTrue(encoded.examples.shape[1] == 5)

        shutil.rmtree(path)

    def test_group_repertoires_3(self):

        path = EnvironmentSettings.root_path + "test/tmp/groupbyannotationsummationstep/"

        PathBuilder.build(path)

        group_columns = ["diabetes"]

        step = GroupDataTransformation(axis=AxisType.REPERTOIRES,
                                       group_summarization_type=GroupSummarizationType.SUM,
                                       group_columns=group_columns,
                                       result_path=path,
                                       filename="encoded_dataset.iml_dataset")

        dataset = step.fit_transform(TestGroupByAnnotationSummation.dataset)

        encoded = dataset.encoded_data

        self.assertTrue(pd.DataFrame(encoded.labels).shape[0] == encoded.examples.shape[0])
        self.assertTrue(len(encoded.example_ids) == encoded.examples.shape[0])
        self.assertTrue(encoded.examples.shape[0] == 2)
        self.assertTrue(encoded.examples.shape[1] == 5)
        self.assertTrue(np.equal(encoded.examples[0, 0], 1))
        self.assertTrue(np.equal(encoded.examples[1, 0], 91))

        shutil.rmtree(path)

    def test_group_features_1(self):
        path = EnvironmentSettings.root_path + "test/tmp/groupbyannotationsummationstep/"

        PathBuilder.build(path)

        group_columns = ["specificity"]

        step = GroupDataTransformation(axis=AxisType.FEATURES,
                                       group_summarization_type=GroupSummarizationType.AVERAGE,
                                       group_columns=group_columns,
                                       result_path=path,
                                       filename="encoded_dataset.iml_dataset")

        dataset = step.fit_transform(TestGroupByAnnotationSummation.dataset)

        encoded = dataset.encoded_data

        self.assertTrue(encoded.examples.shape[1] == encoded.feature_annotations.shape[0])
        self.assertTrue(len(encoded.feature_names) == encoded.feature_annotations.shape[0])
        self.assertTrue(encoded.examples.shape[0] == 5)
        self.assertTrue(encoded.examples.shape[1] == 3)

        shutil.rmtree(path)
        
    def test_group_features_2(self):
        path = EnvironmentSettings.root_path + "test/tmp/groupbyannotationsummationstep/"

        PathBuilder.build(path)

        group_columns = ["specificity", "something"]

        step = GroupDataTransformation(axis=AxisType.FEATURES,
                                       group_summarization_type=GroupSummarizationType.AVERAGE,
                                       group_columns=group_columns,
                                       result_path=path,
                                       filename="encoded_dataset.iml_dataset")

        dataset = step.fit_transform(TestGroupByAnnotationSummation.dataset)

        encoded = dataset.encoded_data

        self.assertTrue(pd.DataFrame(encoded.labels).shape[0] == encoded.examples.shape[0])
        self.assertTrue(len(encoded.example_ids) == encoded.examples.shape[0])
        self.assertTrue(encoded.examples.shape[0] == 5)
        self.assertTrue(encoded.examples.shape[1] == 4)

        shutil.rmtree(path)
