from functools import lru_cache

import numpy as np

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.logging.Logger import log
from source.pairwise_repertoire_comparison.ComparisonDataBatch import ComparisonDataBatch


class ComparisonData:

    @log
    def __init__(self, repertoire_ids: list, comparison_attributes, sequence_batch_size: int = 10000, path: str = None):

        self.path = path
        self.sequence_batch_size = sequence_batch_size
        self.item_count = 0
        self.batch_paths = []
        self.tmp_batch_paths = [self.path + "batch_0.pickle"]
        self.comparison_attributes = comparison_attributes
        self.repertoire_ids = repertoire_ids
        self.batches = []
        self.tmp_batches = []
        self.store_tmp_batch({}, 0)

    def build_matching_fn(self):
        return lambda repertoire: set(zip(*[repertoire.get_attribute(attribute) for attribute in self.comparison_attributes]))

    def get_item_names(self):
        return np.array([item for items in [batch.items for batch in self.batches] for item in items])

    def get_item_vectors(self, repertoire_ids: list = None):
        for batch in self.get_batches(columns=repertoire_ids):
            for item_index in range(batch.shape[0]):
                yield batch[item_index]

    @lru_cache(maxsize=110)
    def get_repertoire_vector(self, identifier: str):
        repertoire_vector = np.zeros(self.item_count)
        for batch_index, batch in enumerate(self.get_batches(columns=[identifier])):
            start = batch_index * self.sequence_batch_size
            end = start + batch.shape[0]
            repertoire_vector[start: end] = batch[:, 0]
        return repertoire_vector

    def get_item_vector(self, index: int):
        batch_index = int(index / self.sequence_batch_size)
        index_in_batch = index - (batch_index * self.sequence_batch_size)
        return self.batches[batch_index].matrix[index_in_batch]

    def get_batches(self, columns: list = None):
        for index in range(len(self.batches)):
            yield self.get_batch(index, columns)

    def get_batch(self, index: int, columns: list = None):
        if columns is not None:
            column_indices = [self.batches[index].repertoire_index_mapping[col] for col in columns]
            return self.batches[index].matrix[:, column_indices]
        else:
            return self.batches[index].matrix

    @log
    def process_dataset(self, dataset: RepertoireDataset):
        extract_fn = self.build_matching_fn()
        for index, repertoire in enumerate(dataset.get_data()):
            self.process_repertoire(repertoire, str(repertoire.identifier), extract_fn)
            print("Repertoire {} ({}/{}) processed.".format(repertoire.identifier, index+1, len(dataset.get_data())))
        self.merge_tmp_batches_to_matrix()

    def merge_tmp_batches_to_matrix(self):

        for index in range(len(self.tmp_batches)):

            batch = self.load_tmp_batch(index)
            matrix = np.zeros((self.sequence_batch_size, len(self.repertoire_ids)), order='F')
            items = []

            for item_index, item in enumerate(batch):
                items.append(item)
                for repertoire_index, repertoire_id in enumerate(self.repertoire_ids):
                    if repertoire_id in batch[item]:
                        matrix[item_index][repertoire_index] = batch[item][repertoire_id]

            df = np.array(matrix[:len(items)], order='F')
            repertoire_index_mapping = {rep_id: ind for ind, rep_id in enumerate(self.repertoire_ids)}

            self.batches.append(ComparisonDataBatch(matrix=df, items=items, repertoire_index_mapping=repertoire_index_mapping))

    @log
    def process_repertoire(self, repertoire, repertoire_id: str, extract_items_fn):
        items = extract_items_fn(repertoire)
        new_items = self.filter_existing_items(items, repertoire_id)
        self.add_items_for_repertoire(new_items, repertoire_id)

    def filter_existing_items(self, items: list, repertoire_id: str) -> list:
        new_items = items
        for batch_index, batch in enumerate(self.get_tmp_batches()):
            new_items = self._remove_existing_items(new_items, batch, batch_index, repertoire_id)
        return new_items

    def _match_items_to_batch(self, items, batch):

        keep = set(items).difference(batch)

        item_to_update = set(items).intersection(batch)
        value = np.ones(len(item_to_update), dtype=np.bool_)

        return keep, value, item_to_update

    def _remove_existing_items(self, new_items: list, batch: dict, batch_index: int, repertoire_id: str) -> list:

        update = {"value": [], "index": []}

        new_items_to_keep, update["value"], update["index"] = self._match_items_to_batch(new_items, batch)

        for index, item in enumerate(update["index"]):
            batch[item][repertoire_id] = update["value"][index]

        self.store_tmp_batch(batch, batch_index)

        return list(new_items_to_keep)

    def store_tmp_batch(self, batch: dict, batch_index: int):

        if len(self.tmp_batches) > batch_index:
            self.tmp_batches[batch_index] = batch
        elif len(self.tmp_batches) == batch_index:
            self.tmp_batches.append(batch)
        else:
            raise KeyError("ComparisonData: batch_index: {} does not exist. tmp_batches length: {}"
                           .format(batch_index, len(self.tmp_batches)))

    def get_tmp_batches(self):
        for i in range(len(self.tmp_batches)):
            yield self.load_tmp_batch(i)

    def load_tmp_batch(self, batch_index: int) -> dict:
        batch = self.tmp_batches[batch_index]
        return batch

    def add_items_for_repertoire(self, items: list, repertoire_id: str):
        last_batch_index = len(self.tmp_batches)-1
        batch = self.load_tmp_batch(last_batch_index)
        item_index = 0
        while len(batch) < self.sequence_batch_size and item_index < len(items):
            batch[items[item_index]] = {repertoire_id: 1}
            item_index += 1

        self.store_tmp_batch(batch, last_batch_index)

        self.item_count += len(items)

        items = items[item_index:]

        while len(items) > 0:
            batch = {item: {repertoire_id: 1} for item in items[:self.sequence_batch_size]}
            last_batch_index += 1
            self.store_tmp_batch(batch, last_batch_index)
            items = items[self.sequence_batch_size:]
