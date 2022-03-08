import datetime
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState
from scripts.specification_util import update_docs_per_mapping


class DatasetExportInstruction(Instruction):
    """
    DatasetExport instruction takes a list of datasets as input, optionally applies preprocessing steps, and outputs the data in specified formats.

    Arguments:

        datasets (list): a list of datasets to export in all given formats

        preprocessing_sequence (list): which preprocessing sequence to use on the dataset(s), this item is optional and does not have to be specified.
        When specified, the same preprocessing sequence will be applied to all datasets.

        exporters (list): a list of formats in which to export the datasets. Valid formats are class names of any non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.

        number_of_processes (int): how many processes to use during repertoire export (not used for sequence datasets)

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dataset_export_instruction: # user-defined instruction name
            type: DatasetExport # which instruction to execute
            datasets: # list of datasets to export
                - my_generated_dataset
                - my_dataset_from_adaptive
            preprocessing_sequence: my_preprocessing_sequence
            number_of_processes: 4
            export_formats: # list of formats to export the datasets to
                - AIRR
                - ImmuneML

    """

    def __init__(self, datasets: List[Dataset], exporters: List[DataExporter], number_of_processes: int = 1,
                 preprocessing_sequence: List[Preprocessor] = None, result_path: Path = None, name: str = None):
        self.datasets = datasets
        self.exporters = exporters
        self.preprocessing_sequence = preprocessing_sequence
        self.result_path = result_path
        self.number_of_processes = number_of_processes
        self.name = name

    def run(self, result_path: Path) -> DatasetExportState:
        self.result_path = result_path / self.name
        paths = {}

        for dataset in self.datasets:
            dataset_name = dataset.name if dataset.name is not None else dataset.identifier

            if self.preprocessing_sequence is not None and len(self.preprocessing_sequence) > 0:
                for preprocessing in self.preprocessing_sequence:
                    dataset = preprocessing.process_dataset(dataset, result_path)
                    print(f"{datetime.datetime.now()}: Preprocessed dataset {dataset_name} with {preprocessing.__class__.__name__}", flush=True)

            paths[dataset_name] = {}
            for exporter in self.exporters:
                export_format = exporter.__name__[:-8]
                path = self.result_path / dataset_name / export_format
                exporter.export(dataset, path, number_of_processes=self.number_of_processes)
                paths[dataset_name][export_format] = path
                contains = str(dataset.__class__.__name__).replace("Dataset", "s").lower()
                print(f"{datetime.datetime.now()}: Exported dataset {dataset_name} containing {dataset.get_example_count()} {contains} in {export_format} format.", flush=True)

        return DatasetExportState(datasets=self.datasets, formats=[exporter.__name__[:-8] for exporter in self.exporters],
                                  preprocessing_sequence=self.preprocessing_sequence, paths=paths, result_path=self.result_path, name=self.name)

    @staticmethod
    def get_documentation():
        doc = str(DatasetExportInstruction.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", "dataset_export/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "Valid formats are class names of any non-abstract class inheriting "
            ":py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.": f"Valid values are: {valid_strategy_values}.",
            "preprocessing_sequence (list)": "preprocessing_sequence (str)",
            "exporters (list)": "formats (list)"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
