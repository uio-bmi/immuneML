import glob
import os
from importlib import import_module

from source.environment.EnvironmentSettings import EnvironmentSettings


class ReflectionHandler:

    @staticmethod
    def get_class_from_path(path: str, class_name: str = None):
        """
        :param path: path to file where class is located
        :param class_name: class name to load from the file; if None, it is assument that the class name is the same
                as the file name
        :return: class
        """
        if class_name is None:
            class_name = os.path.basename(path)[:-3]

        return ReflectionHandler._import_class(path, class_name)

    @staticmethod
    def _import_class(path: str, class_name: str):
        mod = import_module(path[path.rfind("source"):].replace("../", "").replace("/", ".")[:-3])
        cls = getattr(mod, class_name)
        return cls

    @staticmethod
    def get_class_by_name(class_name: str):
        filename = glob.glob(EnvironmentSettings.root_path + "source/**/{}.py".format(class_name))
        if len(filename) != 1:
            raise ValueError("ReflectionHandler could not find class named {}. Check spelling and try again."
                             .format(class_name))
        filename = filename[0]

        return ReflectionHandler._import_class(filename, class_name)
