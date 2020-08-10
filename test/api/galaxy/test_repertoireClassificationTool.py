import os
import shutil
import random as rn
from unittest import TestCase


from source.IO.dataset_export.PickleExporter import PickleExporter
from source.api.galaxy.RepertoireClassificationTool import RepertoireClassificationTool
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestDatasetGenerationTool(TestCase):

    def make_random_dataset(self, path):
        alphabet = EnvironmentSettings.get_sequence_alphabet()
        sequences = ["".join([rn.choice(alphabet) for i in range(20)]) for i in range(100)]

        repertoires, metadata = RepertoireBuilder.build(sequences, path)
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        PickleExporter.export(dataset, path)

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path + "galaxy_repertoire_classification/"
        result_path = path + "result/"
        PathBuilder.build(result_path)

        old_working_dir = os.getcwd()
        os.chdir(path)

        self.make_random_dataset(path)

        args = ['-o', path, '-l', 'donor', '-m', 'RandomForestClassifier', 'SimpleLogisticRegression', 'SVM', 'KNN', '-t', '70', '-c', '5', '-s', 'complete', '-r', 'unique']

        tool = RepertoireClassificationTool(args=args, output_dir=result_path)
        tool.run()

        os.chdir(old_working_dir)

        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split1/"))
        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split2/"))
        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split3/"))
        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split4/"))
        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split5/"))
        self.assertTrue(os.path.exists(f"{result_path}/assessment_random/split_1/selection_random/split_1/donor_e1_ml1"))

        shutil.rmtree(path)
