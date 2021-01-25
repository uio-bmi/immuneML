from unittest import TestCase

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.DatasetExportParser import DatasetExportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction


class TestDatasetExportParser(TestCase):
    def test_parse(self):
        specs = {"type": "DatasetExport", "export_formats": ["Pickle", "AIRR"], "datasets": ["d1"]}

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, RepertoireDataset())

        instruction = DatasetExportParser().parse("instr1", specs, symbol_table)

        self.assertTrue(isinstance(instruction, DatasetExportInstruction))
        self.assertEqual(2, len(instruction.exporters))
        self.assertEqual(1, len(instruction.datasets))
