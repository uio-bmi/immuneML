import shutil

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.SequenceLengthFilter import SequenceLengthFilter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_process_dataset():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'seq_len_filter')

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(5, sequence_count_probabilities={50: 1.},
                                                                 sequence_length_probabilities={3: 0.5, 5: 0.3, 4: 0.2},
                                                                 labels={}, path=path / 'initial_dataset')

    filter = SequenceLengthFilter.build_object(min_len=4, max_len=-1, sequence_type='amino_acid', name='test_sl_filter')

    processed_dataset = filter.process_dataset(dataset, path / 'processed')

    assert len(processed_dataset.repertoires) == 5

    for repertoire in processed_dataset.repertoires:
        assert all(len(s) >= 4 for s in repertoire.get_sequence_aas())

    shutil.rmtree(path)
