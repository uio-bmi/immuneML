from unittest import TestCase

from immuneML.dsl.instruction_parsers.TrainMLModelParser import TrainMLModelParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.config.SplitType import SplitType


class TestTrainMLModelParserSplitConfig(TestCase):

    def test_parse_split_config_without_stratified_k_fold_config(self):
        label_config = LabelConfiguration([Label("disease", [0, 1])])
        instruction = {"split_config": {"split_strategy": "stratified_k_fold", "split_count": 3}}

        split_config = TrainMLModelParser()._parse_split_config("inst1", instruction, "split_config", SymbolTable(), 1, label_config)

        self.assertEqual(SplitType.STRATIFIED_K_FOLD, split_config.split_strategy)
        self.assertIsNone(split_config.stratified_k_fold_config)

    def test_parse_split_config_with_stratified_k_fold_config(self):
        label_config = LabelConfiguration([Label("disease", [0, 1])])
        instruction = {"split_config": {"split_strategy": "stratified_k_fold", "split_count": 3,
                                        "stratified_k_fold_config": {"stratification_label": "subtype"}}}

        split_config = TrainMLModelParser()._parse_split_config("inst1", instruction, "split_config", SymbolTable(), 1, label_config)

        self.assertEqual(SplitType.STRATIFIED_K_FOLD, split_config.split_strategy)
        self.assertEqual("subtype", split_config.stratified_k_fold_config.stratification_label)

    def test_parse_split_config_with_invalid_stratified_k_fold_config_key(self):
        label_config = LabelConfiguration([Label("disease", [0, 1])])
        instruction = {"split_config": {"split_strategy": "stratified_k_fold", "split_count": 3,
                                        "stratified_k_fold_config": {"wrong_key": "subtype"}}}

        with self.assertRaises(AssertionError):
            TrainMLModelParser()._parse_split_config("inst1", instruction, "split_config", SymbolTable(), 1, label_config)

    def test_parse_split_config_with_group_k_fold_config(self):
        label_config = LabelConfiguration([Label("disease", [0, 1])])
        instruction = {"split_config": {"split_strategy": "group_k_fold", "split_count": 3,
                                        "group_k_fold_config": {"group_label": "study", "delimiter": ";",
                                                                "train_only_groups": ["study1"]}}}

        split_config = TrainMLModelParser()._parse_split_config("inst1", instruction, "split_config", SymbolTable(), 1, label_config)

        self.assertEqual(SplitType.GROUP_K_FOLD, split_config.split_strategy)
        self.assertEqual("study", split_config.group_k_fold_config.group_label)
        self.assertEqual(";", split_config.group_k_fold_config.delimiter)
        self.assertEqual(["study1"], split_config.group_k_fold_config.train_only_groups)

    def test_parse_split_config_group_k_fold_without_group_label(self):
        label_config = LabelConfiguration([Label("disease", [0, 1])])
        instruction = {"split_config": {"split_strategy": "group_k_fold", "split_count": 3,
                                        "group_k_fold_config": {"delimiter": ";"}}}

        with self.assertRaises(AssertionError):
            TrainMLModelParser()._parse_split_config("inst1", instruction, "split_config", SymbolTable(), 1, label_config)

