from dataclasses import dataclass, field

from immuneML.util.ParameterValidator import ParameterValidator


@dataclass
class GroupKFoldConfig:

    group_label: str = None
    delimiter: str = None
    train_only_groups: list = field(default_factory=list)

    @staticmethod
    def parse(specs: dict, location: str) -> 'GroupKFoldConfig':
        assert isinstance(specs, dict), f"{location}: group_k_fold_config has to be defined when split_strategy is " \
                                        f"group_k_fold, got {specs} instead."
        ParameterValidator.assert_keys(specs.keys(), ['group_label', 'delimiter', 'train_only_groups'], location,
                                       'group_k_fold_config', exclusive=False)
        ParameterValidator.assert_keys_present(specs.keys(), ['group_label'], location, 'group_k_fold_config')
        ParameterValidator.assert_type_and_value(specs['group_label'], str, location, 'group_label')

        if specs.get('delimiter') is not None:
            ParameterValidator.assert_type_and_value(specs['delimiter'], str, location, 'delimiter')

        train_only_groups = specs.get('train_only_groups') or []
        ParameterValidator.assert_type_and_value(train_only_groups, list, location, 'train_only_groups')

        return GroupKFoldConfig(group_label=specs['group_label'], delimiter=specs.get('delimiter'),
                                train_only_groups=train_only_groups)
