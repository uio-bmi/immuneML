import shutil

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.generative_models.PWM import PWM
from immuneML.reports.gen_model_reports.PWMSummary import PWMSummary
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_pwm_summary():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'pwm_summary')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(10, {10: 0.5, 11: 0.5},
                                                               {}, path / 'dataset')

    pwm = PWM(None, 'amino_acid', 'IMGT_CDR3')

    pwm.fit(dataset, path / 'model')

    report = PWMSummary(dataset, pwm, path / 'report', 'pwm_summary')
    report._generate()

    for file in [path / 'report/length_probs.html', path / 'report/length_probs.csv',
                 path / 'report/pwm_len_10.html', path / 'report/pwm_len_10.csv',
                 path / 'report/pwm_len_11.html', path / 'report/pwm_len_11.csv']:

        assert file.is_file(), file

    shutil.rmtree(path)
