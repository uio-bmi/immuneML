import os
import shutil
import unittest

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.api.galaxy.build_train_gen_model_specs import main as yamlbuilder_main
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_main():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "args_to_yaml")
    data_path = path / "dummy_pickle_data"

    RandomDatasetGenerator.generate_sequence_dataset(path=data_path, labels={'my_label': {1: 0.4, 2: 0.6}},
                                                     sequence_count=10, length_probabilities={5: 1.})

    output_dir = path / "output_dir"
    output_filename = "yaml_out.yaml"

    old_wd = os.getcwd()

    try:
        os.chdir(data_path)

        yamlbuilder_main(["-o", str(output_dir), "-f", output_filename,
                          "-c", "TRA",
                          "-e", "10", "-m", "SoNNia", "-s", "True", "-q", "True",
                          "-k", "True", "-t", "70",
                          "-x", "generated_dataset"])

        # Use ImmuneML parser to test whether the yaml file created here is still valid
        ImmuneMLParser.parse_yaml_file(output_dir / output_filename, path / "result_path")

    finally:
        os.chdir(old_wd)

    shutil.rmtree(path)
