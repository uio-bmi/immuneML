import shutil

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.ShannonDiversityOverview import ShannonDiversityOverview
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_generate():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'shannon_diversity_overview')

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(20, {10: 0.33, 13: 0.07, 15: 0.2, 20: 0.2, 30: 0.2}, {3: 1.},
                                                                 {"celiac": {True: 0.5, False: 0.5},
                                                                  'hla': {'hla1': 0.5, 'hla2': 0.5}}, path / 'dataset')

    report = ShannonDiversityOverview.build_object(result_path=PathBuilder.build(path / 'report'), dataset=dataset,
                                                   name='shannon_diversity_report',
                                                   color_label=None, facet_row_label='celiac', facet_col_label='hla')

    result = report._generate()

    assert all(output.path.is_file() for output in result.output_figures)
    assert all(output.path.is_file() for output in result.output_tables)
    assert len(result.output_figures) == 1
    assert len(result.output_tables) == 1

    shutil.rmtree(path)
