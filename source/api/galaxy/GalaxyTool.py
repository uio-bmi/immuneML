import abc
import os
import shutil
from abc import ABCMeta


class GalaxyTool(metaclass=ABCMeta):

    def __init__(self, specification_path, result_path, **kwargs):
        self.yaml_path = specification_path
        self.result_path = os.path.relpath(result_path) + "/"

    def run(self):
        self._run()
        shutil.make_archive("./immuneML_output", "zip", self.result_path)
        shutil.move("./immuneML_output.zip", self.result_path)

    @abc.abstractmethod
    def _run(self):
        pass
