import shutil

from immuneML.data_model.SequenceParams import RegionType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.signal_annotation import annotate_sequence_dataset
from immuneML.util.PathBuilder import PathBuilder


def test_annotate_sequence_dataset():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "annotated_sequences/")

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {2: 1.}, {'model': {'A': 0.5, 'B': 0.5}}, path)
    signals = [Signal('s1', [SeedMotif('m1', 'A'), SeedMotif('m2', 'C')]), Signal('s2', [SeedMotif('m3', 'G')])]

    annotated_sequences = annotate_sequence_dataset(dataset, signals, region_type=RegionType.IMGT_CDR3,
                                                    sequence_type=SequenceType.AMINO_ACID)

    print(annotated_sequences.head())

    shutil.rmtree(path)
