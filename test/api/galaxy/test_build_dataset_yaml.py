import shutil
import unittest
import yaml


from source.api.galaxy.build_dataset_yaml import main as yamlbuilder_main
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class DummyArguments:
    pass


class MyTestCase(unittest.TestCase):
    def test_main(self):
        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}dataset_yaml/")

        output_dir = f"{path}/output_dir"

        yamlbuilder_main(["-r", "VDJdb", "-o", output_dir, "-f", "receptor.yaml"])
        yamlbuilder_main(["-r", "VDJdb", "-o", output_dir, "-f", "repertoire.yaml", "-m", "metadata.csv"])

        with open(f"{output_dir}/receptor.yaml", "r") as file:
            loaded_receptor = yaml.load(file)

            self.assertDictEqual(loaded_receptor, {"dataset": {"format": "VDJdb", "params": {"path": "./"}}})

        with open(f"{output_dir}/repertoire.yaml", "r") as file:
            loaded_receptor = yaml.load(file)

            self.assertDictEqual(loaded_receptor, {"dataset": {"format": "VDJdb", "params": {"metadata_file": "metadata.csv"}}})

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
