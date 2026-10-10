from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from immuneML.dsl.instruction_parsers.ApplyGenModelParser import ApplyGenModelParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class TestApplyGenModelParser(TestCase):
    def test_parse(self):

        key = "test_instruction"
        instruction = {
            'type': 'some_type',
            'gen_examples_count': 10,
            'reports': ['report_1'],
            'ml_config_path': 'path/to/config.zip'
        }
        symbol_table = SymbolTable()
        report1 = MagicMock()
        base_model = MagicMock()
        expected_model = MagicMock()
        symbol_table.add("report_1", SymbolType.REPORT, report1)
        symbol_table.add("model_name", SymbolType.ML_METHOD, base_model)

        path = Path('path/to/test.zip')

        parser = ApplyGenModelParser()

        expected_instruction = ApplyGenModelInstruction(
            name='test_instruction',
            gen_examples_count=10,
            method=expected_model,
            reports=[symbol_table.get('report_1')]
        )

        parser._load_model = MagicMock(return_value=expected_model)

        result = parser.parse(key, instruction, symbol_table, path)
        self.assertEqual(result.generated_dataset, expected_instruction.generated_dataset)
        self.assertEqual(result.method, expected_instruction.method)
        self.assertEqual(result.reports, expected_instruction.reports)
        self.assertEqual(result.state, expected_instruction.state)

        parser._load_model.assert_called_once_with('path/to/config.zip', 'test_instruction', path)

    def test_parse_with_p_gen_datasets(self):
        instruction = {'type': 'ApplyGenModel', 'gen_examples_count': 10, 'reports': [],
                       'ml_config_path': 'path/to/config.zip', 'p_gen_datasets': ['d1', 'd2']}

        d1, d2 = MagicMock(spec=SequenceDataset), MagicMock(spec=SequenceDataset)
        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, d1)
        symbol_table.add("d2", SymbolType.DATASET, d2)

        model = MagicMock()
        model.can_compute_p_gens.return_value = True
        parser = ApplyGenModelParser()
        parser._load_model = MagicMock(return_value=model)
        path = Path('path/to')

        result = parser.parse("inst", instruction, symbol_table, path)
        self.assertDictEqual({'d1': d1, 'd2': d2}, result.p_gen_datasets)

        self.assertEqual(0, parser.parse("inst", {**instruction, 'gen_examples_count': 0}, symbol_table,
                                         path).state.gen_examples_count)

        model.can_compute_p_gens.return_value = False
        with self.assertRaises(AssertionError):
            parser.parse("inst", instruction, symbol_table, path)
        model.can_compute_p_gens.return_value = True

        for invalid in [{'p_gen_datasets': ['d1', 'missing']}, {'p_gen_datasets': 'd1'},
                        {'p_gen_datasets': [], 'gen_examples_count': 0}]:
            with self.assertRaises(AssertionError):
                parser.parse("inst", {**instruction, **invalid}, symbol_table, path)

        with self.assertRaises(AssertionError):
            parser.parse("inst", {k: v for k, v in instruction.items() if k != 'reports'}, symbol_table, path)

        with self.assertRaises(AssertionError):
            parser.parse("inst", {**instruction, 'p_gen_datasets': None, 'gen_examples_count': 0}, symbol_table, path)
