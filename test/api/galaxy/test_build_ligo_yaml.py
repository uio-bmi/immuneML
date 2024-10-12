import os
import shutil
import unittest

from immuneML.api.galaxy.build_ligo_yaml import main as yamlbuilder_main
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder



class MyTestCase(unittest.TestCase):

    def test_main(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "args_to_yaml")

        output_dir = path / "output_dir"
        output_filename = "yaml_out.yaml"

        old_wd = os.getcwd()

        try:
            os.chdir(path)

            yamlbuilder_main(["-o", str(output_dir), "-f", output_filename, "--motif_seed", "AAA",
                              "--chain_type", "humanTRB", "--signal_percentage", "5",
                              "--example_with_motif_count", "10",  "--example_without_motif_count", "15",
                              "--simulation_strategy", "Implanting", "--dataset_type", "sequence",
                              "--repertoire_size", "20"])

            # Use ImmuneML parser to test whether the yaml file created here is still valid
            ImmuneMLParser.parse_yaml_file(output_dir / output_filename, path / "result_path")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)




if __name__ == '__main__':
    unittest.main()
