from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.preprocessing.filters.Filter import Filter


class ChainRepertoireFilter(Filter):
    """
    Removes all repertoires from the RepertoireDataset object which contain at least one sequence
    with chain different than "keep_chain" parameter.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Since the filter removes repertoires from the dataset (examples in machine learning setting), it cannot be used with :ref:`TrainMLModel`
    instruction. If you want to filter out repertoires including a given chain, see :ref:`DatasetExport` instruction with preprocessing.

    Specification arguments:

    - keep_chain (str): Which chain should be kept, valid values are "TRA", "TRB", "IGH", "IGL", "IGK"


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ChainRepertoireFilter:
                        keep_chain: TRB

    """

    def __init__(self, keep_chain, result_path: Path = None):
        super().__init__(result_path)
        self.keep_chain = Chain.get_chain(keep_chain)

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1):
        self.check_dataset_type(dataset, [RepertoireDataset], "ChainRepertoireFilter")
        processed_dataset = dataset.clone()
        self.result_path = result_path if result_path is not None else self.result_path

        repertoires = []
        indices = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == self.keep_chain for sequence in repertoire.sequences):
                repertoires.append(repertoire)
                indices.append(index)

        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = self._build_new_metadata(processed_dataset, indices)

        self.check_dataset_not_empty(processed_dataset, "ChainRepertoireFilter")

        return processed_dataset

    def keeps_example_count(self) -> bool:
        return False
