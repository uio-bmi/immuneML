import logging
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.preprocessing.filters.Filter import Filter


class CountPerSequenceFilter(Filter):
    """
    Removes all sequences from a Repertoire when they have a count below low_count_limit, or sequences with no count
    value if remove_without_counts is True. This filter can be applied to Repertoires and RepertoireDatasets.

    Specification arguments:

    - low_count_limit (int): The inclusive minimal count value in order to retain a given sequence.

    - remove_without_count (bool): Whether the sequences without a reported count value should be removed.

    - remove_empty_repertoires (bool): Whether repertoires without sequences should be removed.
      Only has an effect when remove_without_count is also set to True. If this is true, this preprocessing cannot be used with :ref:`TrainMLModel`
      instruction, but only with :ref:`DatasetExport` instruction instead.

    - batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed when applying this filter on a RepertoireDataset)


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    CountPerSequenceFilter:
                        remove_without_count: True
                        remove_empty_repertoires: True
                        low_count_limit: 3
                        batch_size: 4

    """

    def __init__(self, low_count_limit: int, remove_without_count: bool, remove_empty_repertoires: bool, batch_size: int, result_path: Path = None):
        super().__init__(result_path)
        self.low_count_limit = low_count_limit
        self.remove_without_count = remove_without_count
        self.remove_empty_repertoires = remove_empty_repertoires
        self.batch_size = batch_size

    def keeps_example_count(self) -> bool:
        return not self.remove_empty_repertoires

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path, number_of_processes=1) -> RepertoireDataset:
        self.check_dataset_type(dataset, [RepertoireDataset], "CountPerSequenceFilter")
        self.result_path = result_path if result_path is not None else self.result_path

        with Pool(self.batch_size) as pool:
            repertoires = pool.map(self._process_repertoire, dataset.repertoires)

        if self.remove_empty_repertoires:
            repertoires = self._remove_empty_repertoires(repertoires)

        processed_dataset = RepertoireDataset.build_from_objects(repertoires=repertoires, path=result_path)

        self.check_dataset_not_empty(processed_dataset, "CountPerSequenceFilter")

        return processed_dataset

    def _process_repertoire(self, repertoire: Repertoire) -> Repertoire:

        counts = repertoire.get_counts()
        counts = counts if counts is not None else np.full(repertoire.get_element_count(), None)
        not_none_indices = counts != None
        counts[not_none_indices] = counts[not_none_indices].astype(int)
        indices_to_keep = np.full(repertoire.get_element_count(), False)
        if self.remove_without_count and self.low_count_limit is not None:
            np.greater_equal(counts, self.low_count_limit, out=indices_to_keep, where=not_none_indices)
        elif self.remove_without_count:
            indices_to_keep = not_none_indices
        elif self.low_count_limit is not None:
            indices_to_keep[np.logical_not(not_none_indices)] = True
            np.greater_equal(counts, self.low_count_limit, out=indices_to_keep, where=not_none_indices)

        processed_repertoire = Repertoire.build_like(repertoire, indices_to_keep, self.result_path, filename_base=f"{repertoire.data_filename.stem}_filtered")

        logging.info(f"{CountPerSequenceFilter.__name__}: finished processing repertoire "
                     f"(subject_id: {repertoire.metadata['subject_id'] if repertoire.metadata and 'subject_id' in repertoire.metadata else ''}, "
                     f"id: {repertoire.identifier}).")

        return processed_repertoire
