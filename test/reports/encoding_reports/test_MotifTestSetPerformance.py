import os
import shutil
from unittest import TestCase

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.MotifTestSetPerformance import MotifTestSetPerformance
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator


class TestMotifTestSetPerformance(TestCase):

    def _get_exported_test_dataset(self, path):
        test_dataset = RandomDatasetGenerator.generate_sequence_dataset(50, {10: 1}, {"is_binder": {"yes": 0.5, "no": 0.5}},
                                                                        path / "test_random_dataset")

        export_path = path / "test_airr_dataset"
        AIRRExporter.export(dataset=test_dataset, path=export_path)

        return export_path

    def _get_encoded_dataset(self, path):
        dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 1}, {"is_binder": {"yes": 0.5, "no": 0.5}},
                                                                   path / "input_dataset")

        lc = LabelConfiguration()
        lc.add_label("is_binder", ["yes", "no"], positive_class="yes")

        encoder = MotifEncoder.build_object(dataset, **{
            "min_positions": 1,
            "max_positions": 1,
            "min_precision": 0.1,
            "min_recall": 0,
            "min_true_positives": 1,
        })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=4,
            learn_model=True,
            model={},
        ))

        return encoded_dataset

    def _write_highlight_motifs_file(self, path):
        file_path = path / "highlight_motifs.tsv"
        with open(file_path, "w") as file:
            file.writelines(["indices\tamino_acids\n", "1\tI\n", "5\tN\n", "0\tA\n", "4&7\t0&1\n"])

        return file_path


    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "motif_test_set_performance/"

        test_dataset_path = self._get_exported_test_dataset(path)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                          "MotifTestSetPerformance")
        params["test_dataset"] = {"format": "AIRR",
                                  "params": {"path": str(test_dataset_path),
                                             "metadata_column_mapping": {"is_binder": "is_binder"}}}
        params["name"] = "motif_set_perf"
        params["highlight_motifs_path"] = str(self._write_highlight_motifs_file(path))

        report = MotifTestSetPerformance.build_object(**params)

        report.dataset = self._get_encoded_dataset(path)
        report.result_path = path / "result_path"

        self.assertTrue(report.check_prerequisites())

        report._generate()

        self.assertTrue(os.path.isfile(path / "result_path/training_set_scores_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result_path/training_combined_precision_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result_path/test_combined_precision_motif_size=1.csv"))
        self.assertTrue(os.path.isfile(path / "result_path/training_precision_per_tp_motif_size=1.html"))
        self.assertTrue(os.path.isfile(path / "result_path/test_precision_per_tp_motif_size=1.html"))

        shutil.rmtree(path)


