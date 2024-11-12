import os
import shutil
from glob import glob

import pytest

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.quickstart import Quickstart


def test_quickstart():
    path = EnvironmentSettings.tmp_test_path / "quickstart_test/"
    PathBuilder.remove_old_and_build(path)

    quickstart = Quickstart()
    quickstart.run(path)

    assert os.path.isfile(path / "machine_learning_analysis/result/full_specs.yaml")
    assert 4 == len(glob(str(path / "machine_learning_analysis/result/machine_learning_instruction/split_1"
                                    "/**/test_predictions.csv"), recursive=True))
    assert os.path.isfile(glob(str(path / "machine_learning_analysis/result/machine_learning_instruction/split_1"
                                          "/**/test_predictions.csv"), recursive=True)[0])

    shutil.rmtree(path, ignore_errors=True)
