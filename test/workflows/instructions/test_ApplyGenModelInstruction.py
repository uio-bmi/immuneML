import shutil
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class TestApplyGenModelInstruction(TestCase):
    def test_run(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "apply_gen_model_instruction/")

        # Create a mock GenerativeModel
        mock_model = Mock()
        mock_model.generate_sequences.return_value = RandomDatasetGenerator.generate_sequence_dataset(2, {2: 1.}, {
            "l1": {"True": 1.}, "l2": {"2": 1.}}, path / "Test/generated_sequences/")

        # Create a mock report
        mock_report = Mock(spec=SequenceLengthDistribution)
        mock_report.generate_report.return_value = "Data Report Result"
        mock_report.name = "rep_name"

        # Create an instance of ApplyGenModelInstruction with the mock objects
        instruction = ApplyGenModelInstruction(method=mock_model, reports=[mock_report],
                                               result_path=path / "generated_sequences/", name="Test",
                                               gen_examples_count=2)

        result = instruction.run(path)

        # Verify that the mock methods were called as expected
        mock_model.generate_sequences.assert_called_with(2, 1, path / "Test/generated_sequences/",
                                                         SequenceType.AMINO_ACID, False)

        # Verify the results
        self.assertEqual(result.name, "Test")
        self.assertEqual(result.result_path, path / "Test/")
        self.assertEqual(result.report_results["data_reports"], ["Data Report Result"])

        df = pd.read_csv(path / "Test/generated_sequences/sequence_dataset.tsv")
        self.assertEqual(2, df.shape[0])

        shutil.rmtree(path)
