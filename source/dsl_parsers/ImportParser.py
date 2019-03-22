import importlib

from source.environment.EnvironmentSettings import EnvironmentSettings


class ImportParser:

    @staticmethod
    def parse(workflow_specification: dict):
        loader = ImportParser.import_loader(workflow_specification)
        dataset = loader.load(workflow_specification["dataset_import"]["path"],
                              {**workflow_specification["dataset_import"]["params"],
                               **{"result_path": workflow_specification["result_path"]}})
        return dataset

    @staticmethod
    def import_loader(workflow_specification: dict):
        loader_path = EnvironmentSettings.root_path + "source/IO/dataset_import/{}Loader".format(workflow_specification["dataset_import"]["format"])
        loader_path = loader_path[loader_path.rfind("source"):].replace("../", "").replace("/", ".")
        module = importlib.import_module(loader_path)
        cls = getattr(module, "{}Loader".format(workflow_specification["dataset_import"]["format"]))
        return cls
