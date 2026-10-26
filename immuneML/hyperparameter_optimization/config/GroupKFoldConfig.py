from dataclasses import dataclass, field
from typing import Union

from immuneML.hyperparameter_optimization.config.SequenceSimilarityConfig import SequenceSimilarityConfig
from immuneML.util.ParameterValidator import ParameterValidator


@dataclass
class GroupKFoldConfig:

    group_label: Union[str, list] = None
    delimiter: str = None
    train_only_groups: list = field(default_factory=list)
    similarity: SequenceSimilarityConfig = None

    def describe_groups(self) -> str:
        if self.similarity is not None:
            return f"similarity: connected components of {self.similarity.describe()}"
        return self.group_label if isinstance(self.group_label, str) else " + ".join(self.group_label)

    @staticmethod
    def parse(specs: dict, location: str) -> 'GroupKFoldConfig':
        assert isinstance(specs, dict), f"{location}: group_k_fold_config has to be defined when split_strategy is " \
                                        f"group_k_fold, got {specs} instead."
        ParameterValidator.assert_keys(specs.keys(), ['group_label', 'delimiter', 'train_only_groups', 'similarity'],
                                       location, 'group_k_fold_config', exclusive=False)
        assert ('group_label' in specs) != ('similarity' in specs), \
            f"{location}: group_k_fold_config has to define exactly one of group_label and similarity, got {specs}."

        if 'similarity' in specs:
            assert specs.get('delimiter') is None and not specs.get('train_only_groups'), \
                (f"{location}: delimiter and train_only_groups can only be used with group_label, since the groups "
                 f"found by similarity have no names to refer to.")
            return GroupKFoldConfig(similarity=SequenceSimilarityConfig.parse(specs['similarity'], location,
                                                                              'group_k_fold_config/similarity',
                                                                              min_edit_distance=1))

        group_label = specs['group_label']
        assert isinstance(group_label, str) or (isinstance(group_label, list) and len(group_label) > 0
                                                and all(isinstance(column, str) for column in group_label)), \
            f"{location}: group_label has to be a column name or a non-empty list of column names, got {group_label}."

        if specs.get('delimiter') is not None:
            ParameterValidator.assert_type_and_value(specs['delimiter'], str, location, 'delimiter')

        train_only_groups = specs.get('train_only_groups') or []
        ParameterValidator.assert_type_and_value(train_only_groups, list, location, 'train_only_groups')

        return GroupKFoldConfig(group_label=specs['group_label'], delimiter=specs.get('delimiter'),
                                train_only_groups=train_only_groups)
