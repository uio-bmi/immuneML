import datetime
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState
from scripts.specification_util import update_docs_per_mapping


class DatasetExportInstruction(Instruction):
    """
    DatasetExport instruction takes a list of datasets as input and outputs them in specified formats.

    Arguments:

        datasets (list): a list of datasets to export in all given formats

        formats (list): a list of formats in which to export the datasets. Valid formats are class names of any non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dataset_export_instruction: # user-defined instruction name
            type: DatasetExport # which instruction to execute
            datasets: # list of datasets to export
                - my_generated_dataset
                - my_dataset_from_adaptive
            export_formats: # list of formats to export the datasets to
                - AIRR
                - Pickle

    """

    def __init__(self, datasets: List[Dataset], exporters: List[DataExporter], result_path: Path = None, name: str = None):
        self.datasets = datasets
        self.exporters = exporters
        self.result_path = result_path
        self.name = name

    def run(self, result_path: Path) -> DatasetExportState:
        self.result_path = result_path / self.name
        paths = {}

        for dataset in self.datasets:
            dataset_name = dataset.name if dataset.name is not None else dataset.identifier
            paths[dataset_name] = {}
            for exporter in self.exporters:
                export_format = exporter.__name__[:-8]
                path = self.result_path / dataset_name / export_format
                exporter.export(dataset, path)
                paths[dataset_name][export_format] = path
                contains = str(dataset.__class__.__name__).replace("Dataset", "s").lower()
                print(f"{datetime.datetime.now()}: Exported dataset {dataset_name} containing {dataset.get_example_count()} {contains} in {export_format} format.", flush=True)

        return DatasetExportState(datasets=self.datasets, formats=[exporter.__name__[:-8] for exporter in self.exporters],
                                      paths=paths, result_path=self.result_path, name=self.name)

    @staticmethod
    def get_documentation():
        doc = str(DatasetExportInstruction.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", "dataset_export/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "Valid formats are class names of any non-abstract class inheriting "
            ":py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.": f"Valid values are: {valid_strategy_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
