from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from immuneML.dsl.instruction_parsers.ApplyGenModelParser import ApplyGenModelParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
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
