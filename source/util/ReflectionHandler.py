import glob
import os
from importlib import import_module

from source.environment.EnvironmentSettings import EnvironmentSettings


class ReflectionHandler:

    @staticmethod
    def import_function(function: str, module):
        return getattr(module, function)

    @staticmethod
    def import_module(name: str, package: str = None):
        return import_module(name, package)

    @staticmethod
    def get_class_from_path(path: str, class_name: str = None):
        """
        :param path: path to file where class is located
        :param class_name: class name to import_dataset from the file; if None, it is assument that the class name is the same
                as the file name
        :return: class
        """
        if class_name is None:
            class_name = os.path.basename(path)[:-3]

        return ReflectionHandler._import_class(path, class_name)

    @staticmethod
    def _import_class(path: str, class_name: str):
        module_path = os.path.normpath(path[path.rfind("source"):]).replace("\\", "/").replace("/", ".")[:-3]
        mod = import_module(module_path)
        cls = getattr(mod, class_name)
        return cls

    @staticmethod
    def get_class_by_name(class_name: str, subdirectory: str = ""):
        filename = glob.glob(EnvironmentSettings.root_path + "source/**/{}.py".format(class_name), recursive=True)
        filename = [f for f in filename if subdirectory in f.replace("\\", "/")]
        if len(filename) != 1:
            raise ValueError("ReflectionHandler could not find class named {}. Check spelling and try again."
                             .format(class_name))
        filename = filename[0]

        return ReflectionHandler._import_class(filename, class_name)

    @staticmethod
    def all_subclasses(cls):
        subclasses = set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in ReflectionHandler.all_subclasses(c)])
        return subclasses

    @staticmethod
    def all_direct_subclasses(cls, drop_part=None, subdirectory=None):
        if drop_part is not None and subdirectory is not None:
            classes = ReflectionHandler.get_classes_by_partial_name(drop_part, subdirectory)
        return [cl for cl in cls.__subclasses__()]

    @staticmethod
    def all_nonabstract_subclass_basic_names(cls, drop_part: str, subdirectory: str = ""):
        return [c.__name__.replace(drop_part, "") for c in ReflectionHandler.all_nonabstract_subclasses(cls, drop_part, subdirectory)]

    @staticmethod
    def all_nonabstract_subclasses(cls, drop_part=None, subdirectory=None):
        if drop_part is not None and subdirectory is not None:
            classes = ReflectionHandler.get_classes_by_partial_name(drop_part, subdirectory)
        return [cl for cl in ReflectionHandler.all_subclasses(cls) if not bool(getattr(cl, "__abstractmethods__", False))]

    @staticmethod
    def exists(class_name: str, subdirectory: str = ""):
        filename = glob.glob(EnvironmentSettings.root_path + "source/**/{}.py".format(class_name), recursive=True)
        filename = [f for f in filename if subdirectory in f.replace("\\", "/")]
        if len(filename) == 1:
            return True
        else:
            return False

    @staticmethod
    def discover_classes_by_partial_name(class_name_ending: str, subdirectory: str = ""):
        filenames = glob.glob(EnvironmentSettings.root_path + "source/**/*{}.py".format(class_name_ending), recursive=True)
        class_names = [f.rpartition("/")[2][:-3] for f in filenames if subdirectory in f.replace("\\", "/")]
        return class_names

    @staticmethod
    def get_classes_by_partial_name(class_name_ending: str, subdirectory: str = ""):
        filenames = glob.glob(EnvironmentSettings.root_path + "source/**/*{}.py".format(class_name_ending), recursive=True)
        filenames = [f for f in filenames if subdirectory in f.replace("\\", "/") and f'{class_name_ending}.py' in f]
        classes = [ReflectionHandler._import_class(filename, os.path.basename(filename)[:-3]) for filename in filenames]
        return classes
