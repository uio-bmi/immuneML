import os
import shutil
import unittest

from immuneML.api.galaxy.build_clustering_yaml import main as yamlbuilder_main
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder



class MyTestCase(unittest.TestCase):
    def create_dummy_dataset(self, path):
        dataset = RandomDatasetGenerator.generate_repertoire_dataset(10, {5: 1}, {4: 1}, {"label1": {True: 0.4, False: 0.6}}, path, name="dataset")

        return f"{dataset.name}.yaml"

    def test_main(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "args_to_yaml_cluster")
        data_path = path / "dummy_pickle_data"

        iml_dataset_name = self.create_dummy_dataset(data_path)

        output_dir = path / "output_dir"
        output_filename = "yaml_out.yaml"


        old_wd = os.getcwd()

        try:
            os.chdir(data_path)

            yamlbuilder_main(["-o", str(output_dir), "-f", output_filename, "-k", "3", "-n", "2", "-d", "PCA", "-l", "label1", "-e", "silhouette_score", "mutual_info_score", "-t", "70"])

            # Use ImmuneML parser to test whether the yaml file created here is still valid
            ImmuneMLParser.parse_yaml_file(output_dir / output_filename, path / "result_path")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
