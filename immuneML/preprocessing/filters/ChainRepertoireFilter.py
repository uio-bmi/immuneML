from pathlib import Path

from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.preprocessing.filters.Filter import Filter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class ChainRepertoireFilter(Filter):
    """
    This filter has two options: it can remove repertoires from the dataset which have any chain other than the
    specified one (e.g., keep  only TRB) or it can remove the sequences from the repertoire which do not have the
    desired chain.

    Since the filter may remove repertoires/sequences from the dataset (examples in machine learning setting), it
    cannot be used with :ref:`TrainMLModel` instruction. If you want to filter out repertoires including a given chain,
    see :ref:`DatasetExport` instruction with preprocessing.

    **Dataset types:**

    - RepertoireDataset

    **Specification arguments:**

    - keep_chains (list): Which chains should be kept, valid values are "TRA", "TRB", "IGH", "IGL", "IGK"

    - remove_only_sequences (bool): Whether to remove only sequences with different chain than "keep_chain" (true) in
      case of repertoire datasets; default is false

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ChainRepertoireFilter:
                        keep_chains: [TRB]
                        remove_only_sequences: true

    """

    def __init__(self, keep_chains: list, remove_only_sequences: bool = False, result_path: Path = None):
        super().__init__(result_path)
        ParameterValidator.assert_type_and_value(keep_chains, list, "ChainRepertoireFilter", "keep_chains")
        self.keep_chains = [Chain.get_chain(keep_chain) for keep_chain in keep_chains]
        self.remove_only_sequences = remove_only_sequences

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1):
        self.check_dataset_type(dataset, [RepertoireDataset], "ChainRepertoireFilter")
        processed_dataset = dataset.clone()
        self.result_path = PathBuilder.build(result_path)
        return self._filter_repertoire_dataset(processed_dataset)

    def _filter_repertoire_dataset(self, processed_dataset: RepertoireDataset):

        repertoires = []
        indices = []
        valid_chains = [chain.value for chain in self.keep_chains]
        for index, repertoire in enumerate(processed_dataset.get_data()):

            if self.remove_only_sequences:
                data = repertoire.data
                data = data[[Chain.get_chain(l).value in valid_chains for l in data.locus.tolist()]]
                if len(data) == 0:
                    continue

                new_repertoire = Repertoire.build_from_dc_object(self.result_path, repertoire.metadata, data=data,
                                                                 filename_base=repertoire.data_filename.stem)
                repertoires.append(new_repertoire)
                indices.append(index)
            else:
                if all(Chain.get_chain(l).value in valid_chains for l in repertoire.data.locus.tolist()):
                    repertoires.append(repertoire)
                    indices.append(index)

        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = self._build_new_metadata(processed_dataset, indices)

        self.check_dataset_not_empty(processed_dataset, "ChainRepertoireFilter")

        return processed_dataset

    def keeps_example_count(self) -> bool:
        return False
