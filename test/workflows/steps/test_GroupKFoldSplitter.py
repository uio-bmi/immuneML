import os
import shutil
from pathlib import Path

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.hyperparameter_optimization.config.GroupKFoldConfig import GroupKFoldConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


def test_group_k_fold_split():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "group_k_fold_splitter/")

    # "big" is always in training; "a;b" belongs to two groups that are held out in different folds, so it is never
    # used, and "a;big" is only used when "a" is not held out
    studies = ["big", "big", "big", "big", "a", "a", "b", "b", "c", "d", "a;b", "a;big"]
    pd.DataFrame({"study": studies, "filename": list(range(len(studies)))}).to_csv(path / "metadata.csv")

    dataset = RepertoireDataset(repertoires=[Repertoire(Path(f"{index}.tsv"), None, str(index))
                                             for index in range(len(studies))],
                                metadata_file=path / "metadata.csv")

    paths = [PathBuilder.build(path / f"split_{i + 1}") for i in range(2)]
    split_config = SplitConfig(split_strategy=SplitType.GROUP_K_FOLD, split_count=2,
                               group_k_fold_config=GroupKFoldConfig(group_label="study", delimiter=";",
                                                                    train_only_groups=["big"]))

    trains, tests = DataSplitter.run(DataSplitterParams(dataset=dataset, split_strategy=SplitType.GROUP_K_FOLD,
                                                        split_count=2, paths=paths, split_config=split_config))

    assert len(trains) == 2 and len(tests) == 2

    held_out_groups = []
    for split_path, train, test in zip(paths, trains, tests):
        train_groups = set(group for value in train.get_metadata(["study"])["study"] for group in value.split(";"))
        test_values = test.get_metadata(["study"])["study"]
        test_groups = set(test_values)

        assert "big" in train_groups
        assert "a;b" not in test_values and "a;big" not in test_values
        assert len(train_groups & test_groups) == 0
        held_out_groups.extend(test_groups)

        assignment = pd.read_csv(split_path / "group_assignment.csv")
        assert set(assignment[assignment["set"] == "test"]["group"]) == test_groups
        assert assignment[assignment["group"] == "big"]["set"].iloc[0] == "train_only"

    # every group except the train-only one is held out exactly once, and the folds are balanced
    assert sorted(held_out_groups) == ["a", "b", "c", "d"]
    assert sorted(test.get_example_count() for test in tests) == [3, 3]

    shutil.rmtree(path)
