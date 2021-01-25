from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter


class ClonesPerRepertoireFilter(Filter):
    """
    Removes all repertoires from the RepertoireDataset, which contain fewer clonotypes than specified by the
    lower_limit, or more clonotypes than specified by the upper_limit.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Arguments:

        lower_limit (int): The minimal inclusive lower limit for the number of clonotypes allowed in a repertoire.

        upper_limit (int): The maximal inclusive upper limit for the number of clonotypes allowed in a repertoire.

    When no lower or upper limit is specified, or the value -1 is specified, the limit is ignored.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ClonesPerRepertoireFilter:
                        lower_limit: 100
                        upper_limit: 100000

    """

    def __init__(self, lower_limit: int = -1, upper_limit: int = -1):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path = None):
        params = {"result_path": result_path}
        if self.lower_limit > -1:
            params["lower_limit"] = self.lower_limit
        if self.upper_limit > -1:
            params["upper_limit"] = self.upper_limit
        return ClonesPerRepertoireFilter.process(dataset, params)

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = dataset.clone()
        repertoires = []
        indices = []
        for index, repertoire in enumerate(dataset.get_data()):
            if "lower_limit" in params.keys() and len(repertoire.sequences) >= params["lower_limit"] or \
                "upper_limit" in params.keys() and len(repertoire.sequences) <= params["upper_limit"]:
                repertoires.append(dataset.repertoires[index])
                indices.append(index)
        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = ClonesPerRepertoireFilter.build_new_metadata(dataset, indices, params["result_path"])

        Filter.check_dataset_not_empty(processed_dataset, "ClonesPerRepertoireFilter")

        return processed_dataset
