import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestRepertoireDataset(TestCase):
    def test_get_metadata_fields(self):

        path = EnvironmentSettings.tmp_test_path / "repertoire_dataset/"
        PathBuilder.build(path)

        repertoires, metadata = RepertoireBuilder.build([["AA"], ["BB"]], path, {"l1": [1, 2], "hla": ["A", "B"]}, subject_ids=["d1", "d2"])
        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)

        self.assertTrue("l1" in dataset.get_metadata_fields())
        self.assertTrue("hla" in dataset.get_metadata_fields())
        self.assertTrue("subject_id" in dataset.get_metadata_fields())

        shutil.rmtree(path)
