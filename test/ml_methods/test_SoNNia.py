import shutil
import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.SoNNia import SoNNia
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_SoNNia():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'sonnia')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 1.},
                                                               {}, path / 'dataset',
                                                               region_type='IMGT_JUNCTION')

    sonnia = SoNNia(batch_size=10000, epochs=5, default_model_name='humanTRB', deep=True,
                    include_joint_genes=True, n_gen_seqs=1000)

    sonnia.fit(dataset, path / 'model')
    sonnia.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/SoNNiaDataset.tsv').exists()
    assert pd.read_csv(str(path / 'generated_dataset/SoNNiaDataset.tsv'), sep='\t').shape[0] == 7

    sonnia.save_model(path)

    sonnia_2 = SoNNia.load_model(path / 'model')
    sonnia_2.generate_sequences(7, 1, path / 'generated_dataset2', SequenceType.AMINO_ACID, False)
    assert (path / 'generated_dataset2').exists()
    assert (path / 'generated_dataset2/SoNNiaDataset.tsv').exists()

    shutil.rmtree(path)
