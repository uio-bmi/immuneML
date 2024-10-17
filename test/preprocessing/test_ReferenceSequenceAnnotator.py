import shutil

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.ReferenceSequenceAnnotator import ReferenceSequenceAnnotator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_process_dataset():

    for compairr_path in EnvironmentSettings.compairr_paths:
        if compairr_path.exists():
            run_test(compairr_path)
            break


def run_test(compairr_path):

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'ref_seq_annotator')

    annotator = ReferenceSequenceAnnotator([ReceptorSequence("AAA", metadata=SequenceMetadata(region_type='FULL_SEQUENCE')),
                                            ReceptorSequence("AAC", metadata=SequenceMetadata(region_type='FULL_SEQUENCE')),
                                            ReceptorSequence("AAT", metadata=SequenceMetadata(region_type='FULL_SEQUENCE')),
                                            ReceptorSequence("AAD", metadata=SequenceMetadata(region_type='FULL_SEQUENCE'))],
                                           0, compairr_path, ignore_genes=True, threads=4, output_column_name='match_test',
                                           repertoire_batch_size=3)

    dataset = RepertoireDataset.build_from_objects(repertoires=RepertoireBuilder.build([['AAA', "AAC", "FCA"],
                                                                                        ["AAT", "ACC"],
                                                                                        ["AAD", "CCC", "AAA"]],
                                                                                       path / 'reps')[0],
                                                   path=path / 'dataset')

    annotated_dataset = annotator.process_dataset(dataset, path / 'result', 4)

    for repertoire in annotated_dataset.repertoires:
        annotations = repertoire.get_attribute('match_test')
        assert annotations is not None
        assert annotations.dtype == int

    shutil.rmtree(path)
