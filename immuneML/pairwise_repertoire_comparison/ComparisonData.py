import logging
import os
import pickle
from pathlib import Path

import numpy as np

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.pairwise_repertoire_comparison.ComparisonDataBatch import ComparisonDataBatch
from immuneML.util.Logger import log
from immuneML.util.PathBuilder import PathBuilder


class ComparisonData:

    @log
    def __init__(self, repertoire_ids: list, comparison_attributes, sequence_batch_size: int = 10000, path: Path = None):

        self.path = PathBuilder.build(path / "comparison_data")
        self.sequence_batch_size = sequence_batch_size
        self.item_count = 0
        self.comparison_attributes = comparison_attributes
        self.repertoire_ids = repertoire_ids
        self.batches = []
        self.tmp_batch_paths = []

    def build_matching_fn(self):
        return lambda repertoire: list(set(zip(*[value for value in repertoire.get_attributes(self.comparison_attributes).values() if value is not None])))

    def get_item_names(self):
        return np.array([item for items in [batch.get_items() for batch in self.batches] for item in items])

    def get_item_vectors(self, repertoire_ids: list = None):
        for batch in self.get_batches(columns=repertoire_ids):
            for item_index in range(batch.shape[0]):
                yield batch[item_index]

    def get_repertoire_vectors(self, identifiers: list):
        repertoire_vectors = {identifier: np.zeros(self.item_count) for identifier in identifiers}
        for batch_index, batch in enumerate(self.get_batches(columns=identifiers, return_dict=True)):
            start = batch_index * self.sequence_batch_size
            for identifier in identifiers:
                end = start + batch[identifier].shape[0]
                repertoire_vectors[identifier][start: end] = batch[identifier]
        return repertoire_vectors

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
        return self.batches[batch_index].get_matrix()[index_in_batch]

    def get_batches(self, columns: list = None, return_dict: bool = False):
        for index in range(len(self.batches)):
            yield self.get_batch(index, columns, return_dict)

    def get_batch(self, index: int, columns: list = None, return_dict: bool = False):
        batch = self.batches[index].load()
        if columns is not None and not return_dict:
            column_indices = [self.batches[index].repertoire_index_mapping[col] for col in columns]
            return batch.get_matrix()[:, column_indices]
        elif columns is not None:
            column_indices = [self.batches[index].repertoire_index_mapping[col] for col in columns]
            matrix = batch.get_matrix()
            return {col: matrix[:, column_indices[i]] for i, col in enumerate(columns)}
        else:
            return batch.get_matrix()

    @log
    def process_dataset(self, dataset: RepertoireDataset):
        extract_fn = self.build_matching_fn()
        repertoire_count = dataset.get_example_count()
        for index, repertoire in enumerate(dataset.get_data()):
            self.process_repertoire(repertoire, str(repertoire.identifier), extract_fn)
            logging.info("Repertoire {} ({}/{}) processed.".format(repertoire.identifier, index+1, repertoire_count))
            logging.info(f"Currently, there are {self.item_count} items in the comparison data matrix.")
        self.merge_tmp_batches_to_matrix()

    def merge_tmp_batches_to_matrix(self):

        for index in range(len(self.tmp_batch_paths)):

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

            comp_data_batch = ComparisonDataBatch(matrix=df, items=items, repertoire_index_mapping=repertoire_index_mapping, path=self.path,
                                                  identifier=index)
            comp_data_batch.store()

            self.batches.append(comp_data_batch)

        for path in self.tmp_batch_paths:
            os.remove(path)

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

        if len(self.tmp_batch_paths) > batch_index or len(self.tmp_batch_paths) == batch_index:
            batch_path = self.path / f'tmp_batch_{batch_index}.pickle'
            with batch_path.open('wb') as file:
                pickle.dump(batch, file)
            if len(self.tmp_batch_paths) == batch_index:
                self.tmp_batch_paths.append(batch_path)
        else:
            raise KeyError("ComparisonData: batch_index: {} does not exist. tmp_batches length: {}"
                           .format(batch_index, len(self.tmp_batch_paths)))

    def get_tmp_batches(self):
        for i in range(len(self.tmp_batch_paths)):
            yield self.load_tmp_batch(i)

    def load_tmp_batch(self, batch_index: int) -> dict:
        if len(self.tmp_batch_paths) > 0 and self.tmp_batch_paths[batch_index].is_file():
            with self.tmp_batch_paths[batch_index].open("rb") as file:
                batch = pickle.load(file)
        else:
            batch = {}
        return batch

    def add_items_for_repertoire(self, items: list, repertoire_id: str):
        last_batch_index = len(self.tmp_batch_paths)-1 if len(self.tmp_batch_paths) > 0 else 0
        batch = self.load_tmp_batch(last_batch_index)

        self.item_count += len(items)

        item_index = 0
        while len(batch) < self.sequence_batch_size and item_index < len(items):
            batch[items[item_index]] = {repertoire_id: 1}
            item_index += 1

        self.store_tmp_batch(batch, last_batch_index)

        items = items[item_index:]

        while len(items) > 0:
            batch = {item: {repertoire_id: 1} for item in items[:self.sequence_batch_size]}
            last_batch_index += 1
            self.store_tmp_batch(batch, last_batch_index)
            items = items[self.sequence_batch_size:]
