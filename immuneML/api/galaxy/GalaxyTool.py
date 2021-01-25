import abc
import os
import shutil
from abc import ABCMeta
from pathlib import Path


class GalaxyTool(metaclass=ABCMeta):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        self.yaml_path = Path(specification_path) if specification_path is not None else None
        self.result_path = Path(os.path.relpath(result_path)) if result_path is not None else None

    def run(self):
        self._run()
        shutil.make_archive(Path("./immuneML_output"), "zip", self.result_path)
        shutil.move(str(Path("./immuneML_output.zip")), str(self.result_path))

    @abc.abstractmethod
    def _run(self):
        pass
