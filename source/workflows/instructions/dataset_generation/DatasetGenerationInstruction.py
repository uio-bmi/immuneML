from typing import List

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.Dataset import Dataset
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.dataset_generation.DatasetGenerationState import DatasetGenerationState


class DatasetGenerationInstruction(Instruction):

    def __init__(self, datasets: List[Dataset], exporters: List[DataExporter], result_path: str = None, name: str = None):
        self.datasets = datasets
        self.exporters = exporters
        self.result_path = result_path
        self.name = name

    def run(self, result_path: str) -> DatasetGenerationState:
        self.result_path = result_path if result_path[-1] == '/' else f"{result_path}/"
        self.result_path = self.result_path + f"{self.name}/"
        paths = {}

        for dataset in self.datasets:
            dataset_name = dataset.name if dataset.name is not None else dataset.identifier
            paths[dataset_name] = {}
            for exporter in self.exporters:
                export_format = exporter.__name__[:-8]
                path = f"{self.result_path}{dataset_name}/{export_format}/"
                exporter.export(dataset, path)
                paths[dataset_name][export_format] = path
                print(f"Exported dataset {dataset_name} in {export_format}.")

        return DatasetGenerationState(datasets=self.datasets, formats=[exporter.__name__[:-8] for exporter in self.exporters],
                                      paths=paths, result_path=self.result_path, name=self.name)
