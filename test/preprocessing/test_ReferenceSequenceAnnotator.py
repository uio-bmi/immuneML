import shutil
from pathlib import Path

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.ReferenceSequenceAnnotator import ReferenceSequenceAnnotator
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_process_dataset():
    compairr_paths = [Path("/usr/local/bin/compairr"), Path("./compairr/src/compairr")]

    for compairr_path in compairr_paths:
        if compairr_path.exists():
            run_test(compairr_path)
            break


def run_test(compairr_path):

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'ref_seq_annotator')

    annotator = ReferenceSequenceAnnotator([ReceptorSequence("AAA", metadata=SequenceMetadata(region_type='IMGT_JUNCTION')),
                                            ReceptorSequence("AAC", metadata=SequenceMetadata(region_type='IMGT_JUNCTION')),
                                            ReceptorSequence("AAT", metadata=SequenceMetadata(region_type='IMGT_JUNCTION')),
                                            ReceptorSequence("AAD", metadata=SequenceMetadata(region_type='IMGT_JUNCTION'))],
                                           0, compairr_path, ignore_genes=True, threads=4, output_column_name='match_test')

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(5, {500: 1.}, {3: 1}, {}, path / 'input_dataset')

    annotated_dataset = annotator.process_dataset(dataset, path / 'result', 4)

    for repertoire in annotated_dataset.repertoires:
        annotations = repertoire.get_attribute('match_test')
        assert annotations is not None
        assert annotations.dtype == int

    shutil.rmtree(path)
