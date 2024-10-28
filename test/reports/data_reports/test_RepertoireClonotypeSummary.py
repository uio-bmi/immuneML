import shutil

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.RepertoireClonotypeSummary import RepertoireClonotypeSummary
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_generate():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'repertoire_clonotype_summary')

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {3: 0.33, 4: 0.07, 5: 0.2, 6: 0.2, 7: 0.2}, {2: 1.},
                                                                 {"celiac": {True: 0.5, False: 0.5}}, path / 'dataset')

    report = RepertoireClonotypeSummary.build_object(result_path=PathBuilder.build(path / 'report'), dataset=dataset, name='test_clonotype_report',
                                                     split_by_label=True, label=None)

    result = report._generate()

    assert all(output.path.is_file() for output in result.output_figures)
    assert all(output.path.is_file() for output in result.output_tables)
    assert len(result.output_figures) == 1
    assert len(result.output_tables) == 1

    shutil.rmtree(path)
