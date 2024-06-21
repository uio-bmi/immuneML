from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter


class ClonesPerRepertoireFilter(Filter):
    """
    Removes all repertoires from the RepertoireDataset, which contain fewer clonotypes than specified by the
    lower_limit, or more clonotypes than specified by the upper_limit.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.
    When no lower or upper limit is specified, or the value -1 is specified, the limit is ignored.

    Since the filter removes repertoires from the dataset (examples in machine learning setting), it cannot be used with :ref:`TrainMLModel`
    instruction. If you want to use this filter, see :ref:`DatasetExport` instruction with preprocessing.

    **Specification arguments:**

    - lower_limit (int): The minimal inclusive lower limit for the number of clonotypes allowed in a repertoire.

    - upper_limit (int): The maximal inclusive upper limit for the number of clonotypes allowed in a repertoire.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ClonesPerRepertoireFilter:
                        lower_limit: 100
                        upper_limit: 100000

    """

    def __init__(self, result_path: Path = None, lower_limit: int = -1, upper_limit: int = -1):
        super().__init__(result_path)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def keeps_example_count(self) -> bool:
        return False

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1):
        self.check_dataset_type(dataset, [RepertoireDataset], "ClonesPerRepertoireFilter")
        self.result_path = result_path if result_path is not None else self.result_path

        repertoires, indices = [], []

        for index, repertoire in enumerate(dataset.get_data()):
            if self.lower_limit != -1 and len(repertoire.sequences) < self.lower_limit:
                continue
            if self.upper_limit != -1 and len(repertoire.sequences) > self.upper_limit:
                continue
            repertoires.append(dataset.repertoires[index])
            indices.append(index)

        processed_dataset = RepertoireDataset.build_from_objects(repertoires=repertoires, path=result_path)

        self.check_dataset_not_empty(processed_dataset, "ClonesPerRepertoireFilter")

        return processed_dataset
