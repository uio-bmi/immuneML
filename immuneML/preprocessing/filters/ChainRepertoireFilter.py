from pathlib import Path

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.PathBuilder import PathBuilder


class ChainRepertoireFilter(Filter):
    """
    Removes all repertoires from the RepertoireDataset object which contain at least one sequence
    with chain different than "keep_chain" parameter.
    Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.

    Arguments:

        keep_chain (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Which chain should be kept.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ChainRepertoireFilter:
                        keep_chain: TRB

    """

    def __init__(self, keep_chain: Chain):
        self.keep_chain = keep_chain

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path = None):
        return ChainRepertoireFilter.process(dataset=dataset, params={"keep_chain": self.keep_chain,
                                                                      "result_path": result_path})

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        processed_dataset = dataset.clone()
        PathBuilder.build(params["result_path"])
        repertoires = []
        indices = []
        for index, repertoire in enumerate(dataset.get_data()):
            if all(sequence.metadata.chain == Chain.get_chain(params["keep_chain"]) for sequence in repertoire.sequences):
                repertoires.append(repertoire)
                indices.append(index)

        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = ChainRepertoireFilter.build_new_metadata(processed_dataset, indices, params["result_path"])

        Filter.check_dataset_not_empty(processed_dataset, "ChainRepertoireFilter")

        return processed_dataset
