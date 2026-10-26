from pathlib import Path

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.hyperparameter_optimization.config.GroupKFoldConfig import GroupKFoldConfig
from immuneML.hyperparameter_optimization.config.ManualSplitConfig import ManualSplitConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.split_dataset.SplitDatasetInstruction import SplitDatasetInstruction, \
    SplitDatasetState


class SplitDatasetParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> SplitDatasetInstruction:

        ParameterValidator.assert_keys(list(instruction.keys()), ['dataset', 'split_config', "type"], "SplitDatasetParser", key)

        dataset = symbol_table.get(instruction["dataset"])
        split_config = self._parse_config(key, instruction)
        state = SplitDatasetState(dataset=dataset, split_config=split_config, name=key)

        return SplitDatasetInstruction(state)


    def _parse_config(self, inst_name: str, instruction: dict):
        split_key = 'split_config'
        try:

            split_strategy = SplitType[instruction[split_key]["split_strategy"].upper()]
            training_percentage = float(
                instruction[split_key]["training_percentage"]) if split_strategy == SplitType.RANDOM else -1

            assert split_strategy in [SplitType.RANDOM, SplitType.MANUAL, SplitType.GROUP_K_FOLD], \
                (f"SplitDatasetParser: this instruction only supports the following split strategies: "
                 f"[RANDOM, MANUAL, GROUP_K_FOLD], got {split_strategy}.")

            if split_strategy == SplitType.GROUP_K_FOLD:
                assert isinstance(instruction[split_key]["split_count"], int) and instruction[split_key]["split_count"] >= 2, \
                    (f"SplitDatasetParser: with split strategy GROUP_K_FOLD, split_count has to be an integer of at "
                     f"least 2, got {instruction[split_key]['split_count']}.")
            else:
                assert instruction[split_key]["split_count"] == 1, \
                    "SplitDatasetParser: with split strategies RANDOM and MANUAL, this instruction only supports " \
                    "splitting the dataset into two (split_count: 1)."

            if split_strategy == SplitType.MANUAL:
                ParameterValidator.assert_keys(keys=instruction[split_key]["manual_config"].keys(),
                                               valid_keys=["train_metadata_path", "test_metadata_path"],
                                               location=SplitDatasetParser.__name__, parameter_name="manual_config",
                                               exclusive=True)

                ParameterValidator.assert_valid_tabular_file(
                    instruction[split_key]["manual_config"]["train_metadata_path"],
                    location=SplitDatasetParser.__name__,
                    parameter_name="train_metadata_path")

                ParameterValidator.assert_valid_tabular_file(
                    instruction[split_key]["manual_config"]["test_metadata_path"],
                    location=SplitDatasetParser.__name__,
                    parameter_name="test_metadata_path")

            return SplitConfig(split_strategy=split_strategy,
                               split_count=int(instruction[split_key]["split_count"]),
                               training_percentage=training_percentage,
                               reports=None,
                               manual_config=ManualSplitConfig(
                                   **instruction[split_key]["manual_config"]) if "manual_config" in instruction[
                                   split_key] else None,
                               leave_one_out_config=None,
                               group_k_fold_config=GroupKFoldConfig.parse(instruction[split_key].get("group_k_fold_config"),
                                                                          f"{SplitDatasetParser.__name__}/{inst_name}")
                               if split_strategy == SplitType.GROUP_K_FOLD else None)

        except KeyError as key_error:
            raise KeyError(
                f"{SplitDatasetParser.__name__}: parameter {key_error.args[0]} was not defined under {split_key} "
                f"in instruction {inst_name}.") from key_error