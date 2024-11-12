import os
import shutil

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.CountPerSequenceFilter import CountPerSequenceFilter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_count_per_seq_filter():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "count_per_seq_filter/")

    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    seq_metadata = [[{"duplicate_count": 1}, {"duplicate_count": 2}, {"duplicate_count": 3}],
                    [{"duplicate_count": 4}, {"duplicate_count": 1}],
                    [{"duplicate_count": 5}, {"duplicate_count": 6}, {"duplicate_count": -1},
                     {"duplicate_count": 1}]]
    dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                     ["ACF", "ACF"],
                                                                     ["ACF", "ACF", "ACF", "ACF"]], path / "dataset1",
                                                                    seq_metadata=seq_metadata)[0])

    dataset1 = CountPerSequenceFilter(**{"low_count_limit": 2, "remove_without_count": True, "remove_empty_repertoires": False,
                                             "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset1'))
    assert 2 == len(dataset1.repertoires[0].data)

    dataset2 = CountPerSequenceFilter(**{"low_count_limit": 5, "remove_without_count": True, "remove_empty_repertoires": False,
                                         "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset2'))
    assert 0 == len(dataset2.repertoires[0].data)

    dataset3 = CountPerSequenceFilter(**{"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
                                         "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset3'))
    assert 3 == len(dataset3.repertoires[2].data)

    dataset4 = CountPerSequenceFilter(
        **{"low_count_limit": 4, "remove_without_count": True, "remove_empty_repertoires": True,
           "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'with_removed_repertoires'))
    assert 2 == dataset4.get_example_count()

    dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                     ["ACF", "ACF"],
                                                                     ["ACF", "ACF", "ACF", "ACF"]], path / "dataset2",
                                                                    seq_metadata=[[{"duplicate_count": -1}, {"duplicate_count": -1}, {"duplicate_count": -1}],
                                                                                  [{"duplicate_count": -1}, {"duplicate_count": -1}],
                                                                                  [{"duplicate_count": -1}, {"duplicate_count": -1}, {"duplicate_count": -1},
                                                                                   {"duplicate_count": -1}]])[0])

    dataset5 = CountPerSequenceFilter(**{"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
                                         "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset5'))
    assert 0 == len(dataset5.repertoires[0].data)
    assert 0 == len(dataset5.repertoires[1].data)
    assert 0 == len(dataset5.repertoires[2].data)

    with pytest.raises(AssertionError):
        CountPerSequenceFilter(**{"low_count_limit": 10, "remove_without_count": True,
                                  "remove_empty_repertoires": True, "result_path": PathBuilder.build(path / 'dataset6'),
                                  "batch_size": 4}).process_dataset(dataset, path)

    shutil.rmtree(path)
