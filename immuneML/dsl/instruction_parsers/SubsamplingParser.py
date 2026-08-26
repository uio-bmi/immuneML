from pathlib import Path

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.Label import Label
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.subsampling.SubsamplingInstruction import SubsamplingInstruction


class SubsamplingParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> SubsamplingInstruction:
        valid_keys = ["type", "dataset", "subsampled_dataset_sizes", "subsampled_repertoire_size", "label",
                      "subsampled_class_distributions"]
        ParameterValidator.assert_keys(instruction.keys(), valid_keys, SubsamplingParser.__name__, key)

        dataset_keys = symbol_table.get_keys_by_type(SymbolType.DATASET)
        ParameterValidator.assert_in_valid_list(instruction['dataset'], dataset_keys, SubsamplingParser.__name__, f'{key}/dataset')

        dataset = symbol_table.get(instruction['dataset'])
        ParameterValidator.assert_type_and_value(instruction['subsampled_dataset_sizes'], list, SubsamplingParser.__name__, f'{key}/subsampled_dataset_sizes')
        ParameterValidator.assert_all_type_and_value(instruction['subsampled_dataset_sizes'], int, SubsamplingParser.__name__,
                                                     f'{key}/subsampled_dataset_sizes', 1, dataset.get_example_count())

        ParameterValidator.assert_type_and_value(instruction['subsampled_repertoire_size'], int,
                                                 SubsamplingParser.__name__, f'{key}/subsampled_repertoire_size',
                                                 nullable=True)

        label, subsampled_class_distributions = self._parse_class_balance(key, instruction, dataset)

        return SubsamplingInstruction(dataset=dataset,
                                      subsampled_repertoire_size=instruction['subsampled_repertoire_size'],
                                      subsampled_dataset_sizes=instruction['subsampled_dataset_sizes'],
                                      label=label,
                                      subsampled_class_distributions=subsampled_class_distributions,
                                      name=key)

    def _parse_class_balance(self, key: str, instruction: dict, dataset):
        distributions = instruction['subsampled_class_distributions']
        label_spec = instruction['label']

        if distributions is None and label_spec is None:
            return None, None

        assert distributions is not None and label_spec is not None, \
            f"{SubsamplingParser.__name__}: for instruction {key}, 'label' and 'subsampled_class_distributions' have to either both be " \
            f"set (to perform class-balanced subsampling) or both be omitted (to perform uniform random subsampling); got " \
            f"label={label_spec} and subsampled_class_distributions={distributions} instead."

        sizes = instruction['subsampled_dataset_sizes']
        ParameterValidator.assert_type_and_value(distributions, list, SubsamplingParser.__name__, f'{key}/subsampled_class_distributions')
        assert len(distributions) == len(sizes), \
            f"{SubsamplingParser.__name__}: for instruction {key}, 'subsampled_class_distributions' has to have as many elements as " \
            f"'subsampled_dataset_sizes' ({len(sizes)}), got {len(distributions)} instead."

        label_config = LabelHelper.create_label_config([label_spec], dataset, SubsamplingParser.__name__, f'{key}/label')
        label = label_config.get_label_object(label_config.get_labels_by_name()[0])

        resolved_distributions = [self._resolve_distribution(key, index, distribution, label)
                                  for index, distribution in enumerate(distributions)]

        self._assert_feasible(key, dataset, label, sizes, resolved_distributions)

        return label, resolved_distributions

    def _resolve_distribution(self, key: str, index: int, distribution: dict, label: Label) -> dict:
        ParameterValidator.assert_type_and_value(distribution, dict, SubsamplingParser.__name__,
                                                 f'{key}/subsampled_class_distributions[{index}]')

        valid_values_str = [str(v) for v in label.values]
        assert all(str(class_value) in valid_values_str for class_value in distribution.keys()), \
            f"{SubsamplingParser.__name__}: for instruction {key}, subsampled_class_distributions[{index}] contains class value(s) not " \
            f"present in label {label.name} (valid values: {label.values}): {list(distribution.keys())}."

        if len(distribution) == 1:
            assert len(label.values) == 2, \
                f"{SubsamplingParser.__name__}: for instruction {key}, subsampled_class_distributions[{index}] specifies a fraction for " \
                f"only one class, which is only supported for binary labels; label {label.name} has {len(label.values)} classes: " \
                f"{label.values}. Please provide fractions for all classes."
            given_class, given_fraction = next(iter(distribution.items()))
            other_class = [v for v in label.values if str(v) != str(given_class)][0]
            distribution = {given_class: given_fraction, other_class: 1 - given_fraction}
        else:
            assert set(str(v) for v in distribution.keys()) == set(valid_values_str), \
                f"{SubsamplingParser.__name__}: for instruction {key}, subsampled_class_distributions[{index}] has to either specify a " \
                f"fraction for a single class (binary labels only) or for all classes of label {label.name} ({label.values}), got " \
                f"{list(distribution.keys())} instead."

        total = sum(distribution.values())
        assert abs(total - 1) < 1e-6, \
            f"{SubsamplingParser.__name__}: for instruction {key}, the fractions in subsampled_class_distributions[{index}] have to sum " \
            f"to 1, got {total} instead ({distribution})."

        return distribution

    def _assert_feasible(self, key: str, dataset, label: Label, sizes: list, distributions: list):
        available_counts = dataset.get_metadata([label.name], return_df=True)[label.name].astype(str).value_counts()

        for index, (size, distribution) in enumerate(zip(sizes, distributions)):
            class_counts = SubsamplingInstruction.compute_class_counts(distribution, size)
            for class_value, count in class_counts.items():
                available = int(available_counts.get(str(class_value), 0))
                assert available >= count, \
                    f"{SubsamplingParser.__name__}: for instruction {key}, subsampled_class_distributions[{index}] requires {count} " \
                    f"examples with {label.name}={class_value} to build a subsampled dataset of size {size}, but only {available} such " \
                    f"examples are available in dataset {dataset.name}."