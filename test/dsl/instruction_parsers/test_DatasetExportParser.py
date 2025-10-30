from unittest import TestCase

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.DatasetExportParser import DatasetExportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.preprocessing.filters.ClonesPerRepertoireFilter import ClonesPerRepertoireFilter
from immuneML.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction


class TestDatasetExportParser(TestCase):
    def test_parse_no_preproc(self):
        specs = {"type": "DatasetExport", "datasets": ["d1"]}

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, RepertoireDataset())

        instruction = DatasetExportParser().parse("instr1", specs, symbol_table)

        self.assertTrue(isinstance(instruction, DatasetExportInstruction))
        self.assertEqual(1, len(instruction.datasets))
        self.assertIsNone(instruction.preprocessing_sequence)

    def test_parse_preproc(self):
        specs = {"type": "DatasetExport", "datasets": ["d1"], "preprocessing_sequence": "p1"}

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, RepertoireDataset())
        symbol_table.add("p1", SymbolType.PREPROCESSING, [ClonesPerRepertoireFilter(lower_limit=-1, upper_limit=-1)])

        instruction = DatasetExportParser().parse("instr1", specs, symbol_table)

        self.assertTrue(isinstance(instruction, DatasetExportInstruction))
        self.assertEqual(1, len(instruction.datasets))
        self.assertEqual(1, len(instruction.preprocessing_sequence))
        self.assertIsInstance(instruction.preprocessing_sequence[0], ClonesPerRepertoireFilter)

