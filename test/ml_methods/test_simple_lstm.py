import shutil

import pandas as pd

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.bnp_util import write_yaml
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.OLGA import OLGA
from immuneML.ml_methods.generative_models.SimpleLSTM import SimpleLSTM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_simple_lstm():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_lstm')

    # seq_path = OLGA.build_object(default_model_name='humanTRB').generate_sequences(500, 1, PathBuilder.build(
    #     path / 'olga_dataset') / 'dataset.tsv',
    #                                                                                SequenceType.AMINO_ACID, False)
    #
    # df = pd.read_csv(seq_path, sep='\t')
    # dataset = SequenceDataset.build_from_objects([ReceptorSequence(**{k: v for k, v in row.to_dict().items()
    #                                                                   if k not in ['region_type', 'frame_type', 'p_gen',
    #                                                                                'from_default_model']})
    #                                               for _, row in df.iterrows()],
    #                                              PathBuilder.build(path / 'olga_dataset_parsed/'),
    #                                              region_type=RegionType.IMGT_JUNCTION)

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {3: 1.},
                                                               {}, path / 'dataset')

    # lstm = SimpleLSTM(Chain.BETA.name, SequenceType.AMINO_ACID.name, 1024, 0.001, 500,
    #                   128, 1, 256, 0.9, region_type=RegionType.IMGT_JUNCTION.name, device='mps', name='lstm_small', window_size=64)
    lstm = SimpleLSTM(Chain.BETA.name, SequenceType.AMINO_ACID.name, 32, 0.001, 5,
                      128, 1, 256, 0.9, region_type=RegionType.IMGT_CDR3.name, device='cpu',
                      name='lstm_small', window_size=5)
    lstm.fit(dataset, path / 'model')

    gen_seq_count = 10
    lstm.generate_sequences(gen_seq_count, 2, path / 'generated', SequenceType.AMINO_ACID, False)

    assert (path / 'generated/synthetic_lstm_dataset.tsv').is_file()
    assert (path / 'generated/synthetic_lstm_dataset.yaml').is_file()

    sequence_df = pd.read_csv(str(path / 'generated/synthetic_lstm_dataset.tsv'), sep='\t')
    assert sequence_df.shape[0] == gen_seq_count
    assert all(sequence_df['cdr3_aa'].str.len() >= 1)

    shutil.rmtree(path)


if __name__ == "__main__":
    test_simple_lstm()
