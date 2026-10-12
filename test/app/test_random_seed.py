import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestRandomSeed(TestCase):
    """The top-level random_seed is applied by ImmuneMLApp before the specification is parsed, so it covers
    randomness drawn while parsing (here: generating a RandomReceptorDataset) as well as in the instructions."""

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _specs(self, path: Path, random_seed=None) -> dict:
        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "RandomReceptorDataset",
                        "params": {
                            "result_path": str(path / "generated"),
                            "receptor_count": 50,
                            "chain_1_length_probabilities": {5: 0.5, 6: 0.5},
                            "chain_2_length_probabilities": {6: 0.5, 7: 0.5},
                            "labels": {"cmv_epitope": {True: 0.5, False: 0.5}}
                        }
                    }
                }
            },
            "instructions": {
                "export": {"type": "DatasetExport", "datasets": ["d1"]}
            },
            "output": {"format": "HTML"}
        }
        if random_seed is not None:
            specs["random_seed"] = random_seed
        return specs

    def _run(self, path: Path, random_seed=None) -> pd.DataFrame:
        PathBuilder.remove_old_and_build(path)
        specs_file = path / "specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(self._specs(path, random_seed), file)

        ImmuneMLApp(specs_file, path / "result").run()

        exported = sorted((path / "result/export/d1/AIRR").glob("*.tsv"))
        self.assertTrue(len(exported) > 0)
        df = pd.concat([pd.read_csv(f, sep="\t") for f in exported], ignore_index=True)
        return df[["locus", "sequence_aa", "cmv_epitope"]]

    def test_same_seed_reproduces(self):
        root = EnvironmentSettings.tmp_test_path / "random_seed_app"

        first = self._run(root / "first", random_seed=2026)
        second = self._run(root / "second", random_seed=2026)
        other = self._run(root / "other", random_seed=7)

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.equals(other))

        with (root / "first/result/full_specs.yaml").open("r") as file:
            full_specs = yaml.safe_load(file)
        self.assertEqual(2026, full_specs["random_seed"])

        shutil.rmtree(root)

    def test_no_seed_runs(self):
        root = EnvironmentSettings.tmp_test_path / "random_seed_app_none"

        df = self._run(root, random_seed=None)
        self.assertEqual(100, df.shape[0])

        with (root / "result/full_specs.yaml").open("r") as file:
            full_specs = yaml.safe_load(file)
        self.assertNotIn("random_seed", full_specs)

        shutil.rmtree(root)

    def test_invalid_seed(self):
        root = EnvironmentSettings.tmp_test_path / "random_seed_app_invalid"

        for invalid in ["abc", -1, 1.5, True, 2 ** 32]:
            with self.subTest(random_seed=invalid):
                PathBuilder.remove_old_and_build(root)
                specs_file = root / "specs.yaml"
                with specs_file.open("w") as file:
                    yaml.dump(self._specs(root, invalid), file)
                with self.assertRaises(AssertionError):
                    ImmuneMLApp(specs_file, root / "result").set_random_seed()

        shutil.rmtree(root)
