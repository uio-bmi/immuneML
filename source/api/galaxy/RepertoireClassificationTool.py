from source.api.galaxy.GalaxyTool import GalaxyTool
from source.api.galaxy.build_yaml_from_arguments import main
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.PathBuilder import PathBuilder


class RepertoireClassificationTool(GalaxyTool):
    def __init__(self, args, result_path):
        self.args = args
        super().__init__(None, result_path)

    def _run(self):
        yaml_path = main(self.args)

        PathBuilder.build(self.result_path)

        app = ImmuneMLApp(yaml_path, self.result_path)
        output_file_path = app.run()

        return output_file_path
