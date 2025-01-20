import shutil

import numpy as np

from immuneML.data_model.SequenceParams import RegionType
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

    annotator = ReferenceSequenceAnnotator([ReceptorSequence(sequence_aa="AAA"),
                                            ReceptorSequence(sequence_aa="AAC"),
                                            ReceptorSequence(sequence_aa="AAT"),
                                            ReceptorSequence(sequence_aa="AAD")],
                                           0, compairr_path, ignore_genes=True, threads=4,
                                           output_column_name='match_test',
                                           repertoire_batch_size=3, region_type=RegionType.IMGT_CDR3)

    dataset = RepertoireDataset.build_from_objects(repertoires=RepertoireBuilder.build([['AAA', "AAC", "FCA"],
                                                                                        ["AAT", "ACC"],
                                                                                        ["AAD", "CCC", "AAA"]],
                                                                                       path / 'reps')[0],
                                                   path=path / 'dataset')

    annotated_dataset = annotator.process_dataset(dataset, path / 'result', 4)

    for ind, repertoire in enumerate(annotated_dataset.repertoires):
        annotations = repertoire.data.match_test
        assert annotations is not None
        assert annotations.dtype == int
        if ind == 0:
            assert np.array_equal(annotations, np.array([1, 1, 0]))
        elif ind == 1:
            assert np.array_equal(annotations, np.array([1, 0]))
        else:
            assert np.array_equal(annotations, np.array([1, 0, 1]))

    shutil.rmtree(path)
