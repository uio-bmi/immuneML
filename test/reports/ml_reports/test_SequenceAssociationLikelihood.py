import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.ProbabilisticBinaryClassifier import ProbabilisticBinaryClassifier
from source.reports.ml_reports.SequenceAssociationLikelihood import SequenceAssociationLikelihood


class TestSequenceAssociationLikelihood(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_generate(self):

        path = EnvironmentSettings.tmp_test_path / "sequence_assoc_likelihood/"

        classifier = ProbabilisticBinaryClassifier(10, 0.1)
        classifier.alpha_0 = 26.7
        classifier.beta_0 = 2814963.8
        classifier.alpha_1 = 4
        classifier.beta_1 = 51820.1
        classifier.class_mapping = {0: "-", 1: "+"}
        classifier.label_name = "CMV"

        report = SequenceAssociationLikelihood(method=classifier, result_path=path)

        report_result = report.generate_report()

        self.assertEqual(1, len(report_result.output_figures))
        self.assertTrue(os.path.isfile(report_result.output_figures[0].path))

        shutil.rmtree(path)
