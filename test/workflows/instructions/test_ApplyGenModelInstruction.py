import shutil
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
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

    def test_run_with_p_gen_datasets(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "apply_gen_model_instruction_p_gen/")

        d1 = RandomDatasetGenerator.generate_sequence_dataset(5, {4: 1.}, {}, path / "dataset1/")
        d2 = RandomDatasetGenerator.generate_sequence_dataset(3, {4: 1.}, {}, path / "dataset2/")

        mock_model = Mock()
        mock_model.generate_sequences.return_value = RandomDatasetGenerator.generate_sequence_dataset(
            2, {2: 1.}, {}, path / "Test/generated_sequences/")
        mock_model.compute_p_gens.side_effect = [np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.6, 0.7, 0.8])]

        instruction = ApplyGenModelInstruction(method=mock_model, reports=[], name="Test", gen_examples_count=2,
                                               p_gen_datasets={'d1': d1, 'd2': d2})
        result = instruction.run(path)

        self.assertEqual(SequenceType.AMINO_ACID, mock_model.compute_p_gens.call_args[0][1])
        self.assertEqual(path / "Test/datasets_with_p_gens", result.exported_datasets['datasets_with_p_gens'])

        df = pd.read_csv(path / "Test/datasets_with_p_gens/d1_p_gens.tsv", sep='\t')
        self.assertTrue(np.allclose([0.1, 0.2, 0.3, 0.4, 0.5], df['p_gen']))
        self.assertListEqual(d1.data.topandas()['cdr3_aa'].tolist(), df['cdr3_aa'].tolist())

        df = pd.read_csv(path / "Test/datasets_with_p_gens/d2_p_gens.tsv", sep='\t')
        self.assertTrue(np.allclose([0.6, 0.7, 0.8], df['p_gen']))

        shutil.rmtree(path)

    def test_run_p_gens_only(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "apply_gen_model_instruction_p_gen_only/")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(3, {4: 1.}, {}, path / "dataset/")

        mock_model = Mock()
        mock_model.compute_p_gens.return_value = np.array([0.1, 0.2, 0.3])
        mock_report = Mock(spec=SequenceLengthDistribution)
        mock_report.name = "rep_name"

        result = ApplyGenModelInstruction(method=mock_model, reports=[mock_report], name="Test", gen_examples_count=0,
                                          p_gen_datasets={'d1': dataset}).run(path)

        mock_model.generate_sequences.assert_not_called()
        self.assertListEqual(['datasets_with_p_gens'], list(result.exported_datasets.keys()))
        self.assertEqual([], result.report_results['data_reports'])
        self.assertEqual(3, pd.read_csv(path / "Test/datasets_with_p_gens/d1_p_gens.tsv", sep='\t').shape[0])

        shutil.rmtree(path)
