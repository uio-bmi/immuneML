from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.hyperparameter_optimization.config.SequenceSimilarityConfig import SequenceSimilarityConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SequenceSimilarityHelper import SequenceSimilarityHelper
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


@dataclass
class SplitDatasetState:
    dataset: Dataset
    split_config: SplitConfig
    name: str = None
    result_path: Path = None
    sequence_overlap: SequenceSimilarityConfig = None
    train_data_paths: List[Path] = field(default_factory=list)
    test_data_paths: List[Path] = field(default_factory=list)
    sequence_overlap_paths: List[Path] = field(default_factory=list)


class SplitDatasetInstruction(Instruction):
    """
    This instruction splits the dataset into two as defined by the instruction parameters. It can be used as a first
    step in clustering to obtain discovery and validation datasets, or to leave out the test dataset for classification.

    For classification, :ref:`TrainMLModel` instruction can be used for more complex data splitting (e.g.,
    nested cross-validation with different data splitting strategies).

    **Specification arguments:**

    - dataset (str): name of the dataset to split, as defined previously in the analysis specification

    - split_config (SplitConfig): the split configuration; the split strategy can be random, manual or group_k_fold.
      For random and manual, split_count has to be 1 and the training and test datasets are exported to `train/` and
      `test/` under the instruction's result folder. For group_k_fold, split_count is the number of folds (at least 2),
      and each split is exported to `split_<i>/train/` and `split_<i>/test/`, together with `group_assignment.csv`
      listing for each group whether it is in the test set, the training set, or always in the training set
      (train_only_groups). See SplitConfig for the details of each split strategy.

    - sequence_overlap (dict): optional, for sequence and receptor datasets. If set, each split also reports how close
      the test examples are to the training examples: for every test example, the smallest number of differences to
      any training example is computed, from 0 (identical) up to `max_edit_distance` (0-2), and `sequence_overlap.csv`
      lists how many test examples are at each distance and the fraction they make up, with the rest counted as
      further away. This shows how much of the test set is (nearly) a copy of the training set, e.g. how much a
      random split of receptor data leaks into the test set. The sequences are compared with CompAIRR, ignoring genes
      and counts. It takes `max_edit_distance` (required), `indels` (whether one difference at distance 1 may be an
      insertion or deletion; CompAIRR allows indels only at distance 1, so distance 2 always means two substitutions;
      default false), `region_type` (default IMGT_CDR3), `chains` (receptor datasets only: which chains to compare,
      e.g. [TRB]; by default all, and a receptor is within a distance only if every compared chain is),
      `compairr_path` and `threads` (default 4). Repertoire datasets are not supported.


    **YAML specification:**

    .. code-block:: yaml

        instructions:
            split_dataset1:
                type: SplitDataset
                dataset: d1
                split_config:
                    split_count: 1
                    split_strategy: random
                    training_percentage: 0.5

            split_dataset_by_study: # each study is in the test set of exactly one of the 3 splits
                type: SplitDataset
                dataset: d1
                split_config:
                    split_count: 3
                    split_strategy: group_k_fold
                    group_k_fold_config:
                        group_label: study_id
                        delimiter: ';' # optional, if an example can belong to several studies
                        train_only_groups: [study1] # optional, groups that are never held out

            split_dataset_by_similarity: # no test sequence within one edit of a training sequence
                type: SplitDataset
                dataset: d1
                split_config:
                    split_count: 2
                    split_strategy: group_k_fold
                    group_k_fold_config:
                        similarity:
                            max_edit_distance: 1
                            indels: true
                sequence_overlap: # optional: how close the test sequences are to the training sequences
                    max_edit_distance: 2
                    compairr_path: /path/to/compairr # optional

    """

    def __init__(self, state: SplitDatasetState):
        assert state.split_config.split_count == 1 or state.split_config.split_strategy == SplitType.GROUP_K_FOLD, \
            (f"SplitDataset instruction can only be used with 1 split count for split strategy "
             f"{state.split_config.split_strategy.name}, got {state.split_config.split_count} instead.")
        self.state = state

    def run(self, result_path: Path) -> SplitDatasetState:
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        split_count = self.state.split_config.split_count
        # a single split is written directly under the result path, as train/ and test/
        paths = [self.state.result_path] if split_count == 1 \
            else [PathBuilder.build(self.state.result_path / f"split_{i + 1}") for i in range(split_count)]

        splitter_params = DataSplitterParams(dataset=self.state.dataset, split_strategy=self.state.split_config.split_strategy,
                                             split_count=split_count,
                                             training_percentage=self.state.split_config.training_percentage,
                                             paths=paths, split_config=self.state.split_config)
        train_datasets, test_datasets = DataSplitter.run(splitter_params)

        for path, train_dataset, test_dataset in zip(paths, train_datasets, test_datasets):
            self.state.train_data_paths.append(path / 'train')
            self.state.test_data_paths.append(path / 'test')

            AIRRExporter.export(train_dataset, path / 'train')
            AIRRExporter.export(test_dataset, path / 'test')

            if self.state.sequence_overlap is not None:
                self.state.sequence_overlap_paths.append(self._write_sequence_overlap(train_dataset, test_dataset, path))

        return self.state

    def _write_sequence_overlap(self, train_dataset: Dataset, test_dataset: Dataset, path: Path) -> Path:
        config = self.state.sequence_overlap
        train_sequences = SequenceSimilarityHelper.get_sequences(train_dataset, config.region_type, config.chains)
        test_sequences = SequenceSimilarityHelper.get_sequences(test_dataset, config.region_type, config.chains)

        distances = SequenceSimilarityHelper.smallest_distance_to_reference(test_sequences, train_sequences,
                                                                            config.max_edit_distance,
                                                                            path / "sequence_overlap",
                                                                            compairr_path=config.compairr_path,
                                                                            threads=config.threads,
                                                                            indels=config.indels)

        counts = [int((distances == distance).sum()) for distance in range(config.max_edit_distance + 1)]
        counts.append(int(distances.isna().sum()))
        overlap = pd.DataFrame({
            'distance_to_training_set': [str(distance) for distance in range(config.max_edit_distance + 1)]
                                        + [f"more than {config.max_edit_distance}"],
            'test_sequences': counts,
            'fraction_of_test_set': [count / max(len(test_sequences), 1) for count in counts]
        })

        overlap_path = path / 'sequence_overlap.csv'
        overlap.to_csv(overlap_path, index=False)
        return overlap_path
