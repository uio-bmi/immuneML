import os
import shutil
from unittest import TestCase

from immuneML.api.galaxy.RepertoireClassificationTool import RepertoireClassificationTool
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestRepertoireClassificationTool(TestCase):

    def make_random_dataset(self, path):
        RandomDatasetGenerator.generate_repertoire_dataset(100, {20: 1.},
                                                           {5: 1.}, name="dataset", path=path, labels={})

    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "galaxy_repertoire_classification/")
        result_path = path / "result/"
        PathBuilder.build(result_path)

        old_working_dir = os.getcwd()

        try:
            os.chdir(path)

            self.make_random_dataset(path)

            args = ['-o', str(path), '-l', 'subject_id', '-m', 'RandomForestClassifier', 'LogisticRegression',
                    '-t', '70', '-c', '2', '-s', 'subsequence', '-p', 'invariant', '-g', 'gapped',
                    '-kl', '1', '-kr', '1', '-gi', '0', '-ga', '1', '-r', 'unique']

            tool = RepertoireClassificationTool(args=args, result_path=result_path)
            tool.run()

        finally:
            os.chdir(old_working_dir)

        self.assertTrue(os.path.exists(result_path / "inst1/split_1/"))
        self.assertTrue(os.path.exists(result_path / "inst1/split_2/"))
        self.assertTrue(os.path.exists(result_path / "inst1/split_1/selection_random/split_1/datasets/"))

        shutil.rmtree(path)
