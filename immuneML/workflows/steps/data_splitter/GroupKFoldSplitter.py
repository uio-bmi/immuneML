import logging

import numpy as np
import pandas as pd

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from immuneML.workflows.steps.data_splitter.Util import Util


class GroupKFoldSplitter:
    """
    Splits the dataset into K folds over groups of examples, where the groups are defined by a metadata field. Each
    group (except those listed as train_only_groups) is held out in exactly one fold; the groups are assigned to
    folds so that the folds have as similar a number of examples as possible.
    """

    @staticmethod
    def split_dataset(input_params: DataSplitterParams):
        config = input_params.split_config.group_k_fold_config
        groups_per_example = GroupKFoldSplitter._get_groups_per_example(input_params.dataset, config.group_label,
                                                                        config.delimiter)
        train_only_groups = set(str(group) for group in config.train_only_groups) if config.train_only_groups else set()
        fold_per_group = GroupKFoldSplitter._assign_groups_to_folds(groups_per_example, train_only_groups,
                                                                    input_params.split_count)

        train_datasets, test_datasets = [], []
        for fold in range(input_params.split_count):
            train_indices, test_indices = GroupKFoldSplitter._get_train_test_indices(groups_per_example,
                                                                                     fold_per_group, fold)
            dropped_count = len(groups_per_example) - len(train_indices) - len(test_indices)
            if dropped_count > 0:
                logging.warning(f"{GroupKFoldSplitter.__name__}: in split {fold + 1}, {dropped_count} examples "
                                f"belong both to held-out groups and to groups in the training set, so they were "
                                f"left out of both the training and the test set.")

            train_datasets.append(Util.make_dataset(input_params.dataset, train_indices, input_params, fold,
                                                    Dataset.TRAIN))
            test_datasets.append(Util.make_dataset(input_params.dataset, test_indices, input_params, fold,
                                                   Dataset.TEST))
            GroupKFoldSplitter._write_group_assignment(groups_per_example, fold_per_group, train_only_groups, fold,
                                                       input_params)

        return train_datasets, test_datasets

    @staticmethod
    def _get_groups_per_example(dataset: Dataset, group_label: str, delimiter: str) -> list:
        try:
            values = dataset.get_metadata([group_label])[group_label]
        except KeyError:
            raise KeyError(f"{GroupKFoldSplitter.__name__}: group_label '{group_label}' was not found in the metadata "
                           f"of dataset {dataset.name}.")

        assert all(value is not None and not pd.isna(value) and str(value) != '' for value in values), \
            f"{GroupKFoldSplitter.__name__}: some examples in dataset {dataset.name} have no value for '{group_label}'."

        if delimiter is not None:
            return [set(str(value).split(delimiter)) for value in values]
        else:
            return [{str(value)} for value in values]

    @staticmethod
    def _assign_groups_to_folds(groups_per_example: list, train_only_groups: set, split_count: int) -> dict:
        group_sizes = pd.Series([group for groups in groups_per_example for group in groups]).value_counts()

        missing_train_only = train_only_groups - set(group_sizes.index)
        if len(missing_train_only) > 0:
            logging.warning(f"{GroupKFoldSplitter.__name__}: train_only_groups {sorted(missing_train_only)} do not "
                            f"occur in the dataset.")

        group_sizes = group_sizes[~group_sizes.index.isin(train_only_groups)]
        assert group_sizes.shape[0] >= split_count, \
            (f"{GroupKFoldSplitter.__name__}: there are {group_sizes.shape[0]} groups that can be held out, which is "
             f"fewer than the number of splits ({split_count}).")

        # shuffling before the stable sort breaks ties between groups of the same size at random; the largest
        # remaining group then always goes to the fold with the fewest examples so far
        group_sizes = group_sizes.iloc[np.random.permutation(group_sizes.shape[0])]
        group_sizes = group_sizes.sort_values(ascending=False, kind="stable")

        fold_sizes = np.zeros(split_count, dtype=int)
        fold_per_group = {}
        for group, size in group_sizes.items():
            fold = int(np.argmin(fold_sizes))
            fold_per_group[group] = fold
            fold_sizes[fold] += size

        return fold_per_group

    @staticmethod
    def _get_train_test_indices(groups_per_example: list, fold_per_group: dict, fold: int):
        train_indices, test_indices = [], []
        for index, groups in enumerate(groups_per_example):
            held_out = [fold_per_group.get(group) == fold for group in groups]
            if all(held_out):
                test_indices.append(index)
            elif not any(held_out):
                train_indices.append(index)

        assert len(train_indices) > 0 and len(test_indices) > 0, \
            (f"{GroupKFoldSplitter.__name__}: split {fold + 1} has {len(train_indices)} training and "
             f"{len(test_indices)} test examples; both have to be non-empty.")

        return train_indices, test_indices

    @staticmethod
    def _write_group_assignment(groups_per_example: list, fold_per_group: dict, train_only_groups: set, fold: int,
                                input_params: DataSplitterParams):
        group_sizes = pd.Series([group for groups in groups_per_example for group in groups]).value_counts()
        df = pd.DataFrame({'group': group_sizes.index, 'example_count': group_sizes.values})
        df['set'] = ['train_only' if group in train_only_groups
                     else 'test' if fold_per_group[group] == fold else 'train' for group in df['group']]
        df.to_csv(input_params.paths[fold] / 'group_assignment.csv', index=False)
