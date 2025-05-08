import shutil

from immuneML.data_model.SequenceParams import RegionType, Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.dim_reduction.PCA import PCA
from immuneML.ml_methods.generative_models.SimpleVAE import SimpleVAE
from immuneML.reports.gen_model_reports.VAESummary import VAESummary
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_vae_summary():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'simple_vae_summary')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=100, length_probabilities={10: 1.},
                                                               labels={}, path=path / 'dataset',
                                                               region_type=RegionType.IMGT_JUNCTION.name)

    vae = SimpleVAE(Chain.get_chain('beta'), 0.75, 8, 4, 10, 100, 2,
                    1, 1, 21, 2, 5, 10, 'cpu', 0.01)

    vae.fit(dataset, path / 'model')

    report = VAESummary(2, 4, dataset, vae, path / 'report', 'summary',
                        dim_red_method=PCA(n_components=2))
    report._generate()

    for file in [path / 'report/latent_dim_dist.html', path / 'report/latent_space_PCA.csv',
                 path / 'report/latent_space_PCA.html', path / 'report/loss_per_epoch.html',
                 path / 'report/training_losses.csv']:

        assert file.is_file(), file

    shutil.rmtree(path)
