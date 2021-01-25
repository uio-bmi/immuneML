import os
import random as rn
import shutil
from unittest import TestCase

from immuneML.IO.dataset_export.PickleExporter import PickleExporter
from immuneML.api.galaxy.RepertoireClassificationTool import RepertoireClassificationTool
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestRepertoireClassificationTool(TestCase):

    def make_random_dataset(self, path):
        alphabet = EnvironmentSettings.get_sequence_alphabet()
        sequences = [["".join([rn.choice(alphabet) for i in range(20)]) for i in range(100)] for i in range(40)]

        repertoires, metadata = RepertoireBuilder.build(sequences, path, subject_ids=[i % 2 for i in range(len(sequences))])
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        PickleExporter.export(dataset, path)

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "galaxy_repertoire_classification/"
        result_path = path / "result/"
        PathBuilder.build(result_path)

        old_working_dir = os.getcwd()
        os.chdir(path)

        self.make_random_dataset(path)

        args = ['-o', str(path), '-l', 'subject_id', '-m', 'RandomForestClassifier', 'LogisticRegression',
                '-t', '70', '-c', '2', '-s', 'subsequence', '-p', 'invariant', '-g', 'gapped',
                '-kl', '1', '-kr', '1', '-gi', '0', '-ga', '1', '-r', 'unique']

        tool = RepertoireClassificationTool(args=args, result_path=result_path)
        tool.run()

        os.chdir(old_working_dir)

        self.assertTrue(os.path.exists(result_path / "inst1/split_1/"))
        self.assertTrue(os.path.exists(result_path / "inst1/split_2/"))
        self.assertTrue(os.path.exists(result_path / "inst1/split_1/selection_random/split_1/datasets/"))

        shutil.rmtree(path)
