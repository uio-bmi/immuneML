import abc
import os
import shutil
from abc import ABCMeta
from pathlib import Path
import glob
from immuneML.presentation.html.FailedGalaxyHTMLBuilder import FailedGalaxyHTMLBuilder


class GalaxyTool(metaclass=ABCMeta):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        self.yaml_path = Path(specification_path) if specification_path is not None else None
        self.result_path = Path(os.path.relpath(result_path)) if result_path is not None else None

    def run(self):
        try:
            self._run()
        except Exception as e:
            print("exception")
            self._make_failed_galaxy_run_html()
            raise e
        finally:
            print(glob.glob(str(self.result_path / "*")))
            print("finally")
            shutil.make_archive(Path("./immuneML_output"), "zip", self.result_path)
            shutil.move(str(Path("./immuneML_output.zip")), str(self.result_path))
            print(glob.glob(str(self.result_path / "*")))

    @abc.abstractmethod
    def _run(self):
        pass

    def _make_failed_galaxy_run_html(self):
        print("make failed galaxy run html")
        FailedGalaxyHTMLBuilder.build(self.result_path)

