import random
import shutil
import string
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.sequencing_depth_overview.SequencingDepthOverview import SequencingDepthOverview
from source.util.PathBuilder import PathBuilder


class TestSequencingDepthOverview(TestCase):
    def test_generate(self):

        path = EnvironmentSettings.root_path + "test/tmp/datareports/"
        PathBuilder.build(path)

        repertoires = [SequenceRepertoire.build_from_sequence_objects(self.generate_sequences(),
                                                                      metadata={"disease": random.choice(["t1d", "lupus", "ra", "ms"]),
                                                                                "week": "week" + str(random.randint(0, 4))}, path=path)
                       for i in range(5)]

        dataset = RepertoireDataset(repertoires=repertoires, params={"disease": ["t1d", "lupus", "ra", "ms"],
                                                                     "week": ["week0", "week1", "week2", "week3", "week4"]})

        report = SequencingDepthOverview(dataset, batch_size=1, x="disease", result_path=path, height_distributions=5,
                                         height_scatterplot=2.5, palette={"lupus": "brown", "t1d": "purple", "ms": "purple"})
        report.generate()

        shutil.rmtree(path)

    def generate_sequences(self):
        sequences = [ReceptorSequence(amino_acid_sequence=self.random_string(10),
                                      metadata=SequenceMetadata(count=random.randint(1, 100),
                                                                frame_type=random.choice(list(SequenceFrameType)).name)) for i in range(10)]
        return sequences

    def random_string(self, stringLength=10):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))
