import copy

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.preprocessing.filters.Filter import Filter


class ClonotypeCountFilter(Filter):

    def __init__(self, lower_limit: int = -1, upper_limit: int = -1):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def process_dataset(self, dataset: RepertoireDataset, result_path: str = None):
        params = {"result_path": result_path}
        if self.lower_limit > -1:
            params["lower_limit"] = self.lower_limit
        if self.upper_limit > -1:
            params["upper_limit"] = self.upper_limit
        return ClonotypeCountFilter.process(dataset, params)

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = copy.deepcopy(dataset)
        filenames = []
        indices =[]
        for index, repertoire in enumerate(dataset.get_data()):
            if "lower_limit" in params.keys() and len(repertoire.sequences) >= params["lower_limit"] or \
                 "upper_limit" in params.keys() and len(repertoire.sequences) <= params["upper_limit"]:
                filenames.append(dataset.get_filenames()[index])
                indices.append(index)
        processed_dataset.set_filenames(filenames)
        processed_dataset.metadata_file = ClonotypeCountFilter.build_new_metadata(dataset, indices, params["result_path"])
        return processed_dataset
