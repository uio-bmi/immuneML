import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.encoding_reports.MotifPrecisionTP import MotifPrecisionTP
from immuneML.util.PathBuilder import PathBuilder


class TestMotifPrecisionTP(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_data(self, path, add_weights=False):
        sequences = [
            ReceptorSequence(
                amino_acid_sequence="AACC",
                identifier="1",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                amino_acid_sequence="AGDD",
                identifier="2",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                amino_acid_sequence="AAEE",
                identifier="3",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                amino_acid_sequence="AGFF",
                identifier="4",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                amino_acid_sequence="CCCC",
                identifier="5",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                amino_acid_sequence="DDDD",
                identifier="6",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                amino_acid_sequence="EEEE",
                identifier="7",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                amino_acid_sequence="FFFF",
                identifier="8",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
        ]

        PathBuilder.build(path)

        dataset = SequenceDataset.build_from_objects(
            sequences, 100, PathBuilder.build(path / "data"), "d1"
        )

        if add_weights:
            dataset.set_example_weights([1/i for i in range(1, dataset.get_example_count()+1)])


        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = MotifEncoder.build_object(
            dataset,
            **{
                "max_positions": 3,
                "min_precision": 0.5,
                "min_recall": 0.0,
                "min_true_positives": 1,
                "generalize_motifs": False,
            }
        )

        encoded_dataset = encoder.encode(
            dataset,
            EncoderParams(
                result_path=path / "encoded_data/",
                label_config=lc,
                pool_size=2,
                learn_model=True,
                model={},
            ),
        )

        return encoded_dataset

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_precision_tp/"
        PathBuilder.build(path)

        highlight_motifs_path = path / "motifs_to_highlight.csv"
        with open(highlight_motifs_path, "w") as file:
            file.write("indices	amino_acids\n1	A\n1	G\n0&1	A&A\n0&1	A&G\n0&2	A&C")

        encoded_dataset = self._create_dummy_encoded_data(path)

        report = MotifPrecisionTP.build_object(
            **{"dataset": encoded_dataset,
               "result_path": path,
               "highlight_motifs_path": str(highlight_motifs_path)}
        )

        self.assertTrue(report.check_prerequisites())

        result = report.generate_report()

        self.assertListEqual(report.highlight_motifs, ['1-A', '1-G', '0&1-A&A', '0&1-A&G', '0&2-A&C'])
        self.assertTrue(os.path.isfile(path / "motif_precision_recall_tp.csv"))
        self.assertTrue(os.path.isfile(path / "precision_per_tp.html"))
        self.assertTrue(os.path.isfile(path / "precision_recall.html"))

        shutil.rmtree(path)
