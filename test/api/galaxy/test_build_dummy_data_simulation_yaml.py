import os
import shutil
import unittest

import yaml

from immuneML.api.galaxy.build_dummy_data_simulation_yaml import main as yamlbuilder_main
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class MyTestCase(unittest.TestCase):

    def test_sequencedataset(self):
        path = EnvironmentSettings.tmp_test_path / "sequencedataset_yaml_overview/"
        PathBuilder.remove_old_and_build(path)

        old_wd = os.getcwd()

        try:
            os.chdir(path)

            yamlbuilder_main(["--dataset_type", "sequence", "--output_path", str(path), "--file_name", "dummy_seqs.yaml",
                              "--class_balance", "60", "--label_name", "is_binder", "--class1_name", "yes", "--class2_name", "no",
                              "--count", "20"])

            with open(path / "dummy_seqs.yaml", "r") as file:
                loaded_specs = yaml.load(file, Loader=yaml.FullLoader)

                self.assertDictEqual(loaded_specs["definitions"]["datasets"], {"dataset":
                                                                                   {"format": "RandomSequenceDataset",
                                                                                    "params":
                                                                                        {"sequence_count": 20,
                                                                                         "labels": {"is_binder": {"yes": 0.6, "no": 0.4}}}}})

            ImmuneMLParser.parse_yaml_file(path / "dummy_seqs.yaml", result_path=path)
        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
