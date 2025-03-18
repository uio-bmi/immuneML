from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams


class Util:

    @staticmethod
    def prepare_path(input_params: DataSplitterParams, split_index: int, dataset_type: str) -> str:
        path = input_params.paths[split_index] / f"datasets/{dataset_type.lower()}/"
        PathBuilder.build(path)
        return path

    @staticmethod
    def make_dataset(dataset: Dataset, indices, input_params: DataSplitterParams, i: int, dataset_type: str):
        path = Util.prepare_path(input_params, i, dataset_type)
        new_dataset = dataset.make_subset(indices, path, dataset_type)
        return new_dataset
