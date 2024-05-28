import os
import shutil
import unittest

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.api.galaxy.build_train_gen_model_specs import main as yamlbuilder_main
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder



class MyTestCase(unittest.TestCase):
    def create_dummy_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path, labels={"label1": ["val1", "val2"], "label2": ["val1", "val2"]})

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        dataset.name = "my_dataset"
        ImmuneMLExporter.export(dataset, path)

        return f"{dataset.name}.yaml"

    def test_main(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "args_to_yaml")
        data_path = path / "dummy_pickle_data"

        iml_dataset_name = self.create_dummy_dataset(data_path)

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




if __name__ == '__main__':
    unittest.main()
