import random
import shutil
from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.subsampling.SubsamplingState import SubsamplingState
from scripts.specification_util import update_docs_per_mapping


class SubsamplingInstruction(Instruction):
    """
    Subsampling is an instruction that subsamples a given dataset and creates multiple smaller dataset according to the parameters provided.

    Arguments:

        dataset (Dataset): original dataset which will be used as a basis for subsampling

        subsampled_dataset_sizes (list): a list of dataset sizes (number of examples) each subsampled dataset should have

        dataset_export_formats (list): in which formats to export the subsampled datasets. Valid formats are class names of any non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_subsampling_instruction: # user-defined name of the instruction
            type: Subsampling # which instruction to execute
            dataset: my_dataset # original dataset to be subsampled, with e.g., 300 examples
            subsampled_dataset_sizes: # how large the subsampled datasets should be, one dataset will be created for each list item
                - 200 # one subsampled dataset with 200 examples (200 repertoires if my_dataset was repertoire dataset)
                - 100 # the other subsampled dataset will have 100 examples
            dataset_export_formats: # in which formats to export the subsampled datasets
                - Pickle
                - AIRR

    """

    def __init__(self, dataset: Dataset, subsampled_dataset_sizes: List[int], dataset_export_formats: list, result_path: Path = None, name: str = None):
        self.state = SubsamplingState(dataset, subsampled_dataset_sizes, dataset_export_formats, result_path, name)

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        example_indices = list(range(self.state.dataset.get_example_count()))

        for index, dataset_size in enumerate(self.state.subsampled_dataset_sizes):

            new_dataset_name = f"{self.state.dataset.name}_{dataset_size}_subsampled_{index+1}"
            new_dataset_path = PathBuilder.build(self.state.result_path / new_dataset_name)

            new_example_indices = random.sample(example_indices, k=dataset_size)
            new_dataset = self.state.dataset.make_subset(new_example_indices, new_dataset_path, Dataset.SUBSAMPLED)
            new_dataset.name = new_dataset_name

            self.state.subsampled_datasets.append(new_dataset)

            self.export_dataset(new_dataset, new_dataset_path)

        return self.state

    def export_dataset(self, new_dataset, new_dataset_path):

        self.state.subsampled_dataset_paths[new_dataset.name] = {}

        for exporter in self.state.dataset_exporters:
            exporter_name = exporter.__name__[:-8].lower()
            export_path = new_dataset_path / f"exported/{exporter_name}/"
            exporter.export(new_dataset, export_path)
            zip_export_path = shutil.make_archive(new_dataset_path / f"exported_{exporter_name}_{new_dataset.name}", "zip", export_path)
            self.state.subsampled_dataset_paths[new_dataset.name][exporter_name] = zip_export_path

    @staticmethod
    def get_documentation():
        doc = str(SubsamplingInstruction.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", "dataset_export/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "Valid formats are class names of any non-abstract class inheriting "
            ":py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.": f"Valid values are: {valid_strategy_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
