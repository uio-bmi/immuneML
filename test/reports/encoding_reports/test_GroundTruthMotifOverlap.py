import os
from unittest import TestCase
import pandas as pd
import shutil

from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.GroundTruthMotifOverlap import GroundTruthMotifOverlap
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestGroundTruthMotifOverlap(TestCase):

    def _get_encoded_dataset(self, path):
        dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 1}, {"is_binder": {"yes": 0.5, "no": 0.5}},
                                                                   path / "input_dataset")

        lc = LabelConfiguration()
        lc.add_label("is_binder", ["yes", "no"], positive_class="yes")

        encoder = MotifEncoder.build_object(dataset, **{
            "min_positions": 1,
            "max_positions": 3,
            "min_precision": 0.1,
            "min_recall": 0,
            "min_true_positives": 1,
            "generalize_motifs": False,
        })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoder_result/",
            label_config=lc,
            pool_size=4,
            learn_model=True,
            model={},
        ))

        return encoded_dataset

    def _write_groundtruth_motifs_file(self, path):
        file_path = path / "gt_motifs.tsv"
        with open(file_path, "w") as file:
            file.writelines(["indices\tamino_acids\tn_sequences\n", "1\tI\t6\n", "5\tN\t10\n", "0\tA\t4\n", "4&7\t0&1\t30\n"])

        return file_path

    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "motif_test_set_performance/")

        report = GroundTruthMotifOverlap.build_object(**{"groundtruth_motifs_path": str(self._write_groundtruth_motifs_file(path))})

        report.dataset = self._get_encoded_dataset(path)
        report.result_path = path / "result_path"

        self.assertTrue(report.check_prerequisites())

        report._generate()

        self.assertTrue(os.path.isfile(path / "result_path/ground_truth_motif_overlap.tsv"))

        df = pd.read_csv(path / "result_path/ground_truth_motif_overlap.tsv", sep="\t")

        if len(df) > 0:
            self.assertTrue(os.path.isfile(path / "result_path/ground_truth_motif_overlap.html"))

        shutil.rmtree(path)


