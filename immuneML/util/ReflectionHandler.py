from importlib import import_module
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings


class ReflectionHandler:

    @staticmethod
    def import_function(function: str, module):
        return getattr(module, function)

    @staticmethod
    def import_module(name: str, package: str = None):
        return import_module(name, package)

    @staticmethod
    def get_class_from_path(path, class_name: str = None):
        """ obtain the class reference from the given path

        Args:

            path (str or pathlib.Path): path to file where the class is located
            class_name (str): class name to import_dataset from the file; if None, it is assumed that the class name is the same as the file name

        Returns:
             class
        """
        path = Path(path)
        if class_name is None:
            class_name = path.stem

        return ReflectionHandler._import_class(path, class_name)

    @staticmethod
    def _import_class(path: Path, class_name: str):
        module_path = ".".join(path.parts[len(list(path.parts)) - list(path.parts)[::-1].index("immuneML") - 1:])[:-3]
        mod = import_module(module_path)
        cls = getattr(mod, class_name)
        return cls

    @staticmethod
    def get_class_by_name(class_name: str, subdirectory: str = ""):
        filenames = ReflectionHandler._get_filenames(class_name, subdirectory)

        assert len(filenames) == 1, f"ReflectionHandler could not find class named {class_name}. Check spelling and try again."

        return ReflectionHandler._import_class(filenames[0], class_name)

    @staticmethod
    def _get_filenames(class_name: str, subdirectory_name: str = "", partial=False):
        pattern = f"immuneML/**/*{class_name}.py" if partial else f"immuneML/**/{class_name}.py"

        filenames = list(EnvironmentSettings.root_path.glob(pattern))
        filenames = [f for f in filenames if subdirectory_name in "/".join(f.parts)]

        return filenames

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
        filenames = ReflectionHandler._get_filenames(class_name, subdirectory)

        if len(filenames) == 1:
            return True
        else:
            return False

    @staticmethod
    def discover_classes_by_partial_name(class_name_ending: str, subdirectory: str = ""):
        filenames = ReflectionHandler._get_filenames(class_name_ending, subdirectory, partial=True)
        class_names = [f.stem for f in filenames]

        return class_names

    @staticmethod
    def get_classes_by_partial_name(class_name_ending: str, subdirectory: str = ""):
        filenames = ReflectionHandler._get_filenames(class_name_ending, subdirectory, partial=True)
        classes = [ReflectionHandler._import_class(filename, filename.stem) for filename in filenames]
        return classes
