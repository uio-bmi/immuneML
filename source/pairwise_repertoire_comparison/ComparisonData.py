import pickle
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class ComparisonData:

    def __init__(self, repertoire_ids: list, matching_columns, item_columns: list, pool_size: int, batch_size: int = 10000,
                 path: str = None):

        self.path = path
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.item_count = 0
        self.batch_paths = []
        self.tmp_batch_paths = [self.path + "batch_0.pickle"]
        self.store_tmp_batch({}, 0)
        self.item_columns = item_columns
        self.matching_columns = matching_columns
        self.repertoire_ids = repertoire_ids

    def get_repertoire_vector(self, identifier: str):
        repertoire_vector = np.zeros(self.item_count)
        for batch_index, batch in enumerate(self.get_batches(columns=[identifier])):
            part = batch[identifier].values
            start = batch_index * self.batch_size
            end = start + part.shape[0]
            repertoire_vector[start: end] = part
        return repertoire_vector

    def get_item_vector(self, index: int):
        batch_index = int(index / self.batch_size)
        index_in_batch = index - (batch_index * self.batch_size)
        return pd.read_csv(self.batch_paths[batch_index], nrows=1, skiprows=index_in_batch - 1).iloc[0].values

    def get_batches(self, columns: list = None):
        for batch_path in self.batch_paths:
            yield self.get_batch(batch_path, columns)

    def get_batch(self, batch_path: str, columns: list = None):
        if columns is not None:
            return pd.read_csv(batch_path, usecols=columns)
        else:
            return pd.read_csv(batch_path)

    def process_dataset(self, dataset: RepertoireDataset, extract_items_fn):
        for index, repertoire in enumerate(dataset.get_data()):
            self.process_repertoire(repertoire, str(repertoire.identifier), extract_items_fn)
        self.merge_tmp_batches_to_matrix()

    def merge_tmp_batches_to_matrix(self):
        for index, path in enumerate(self.tmp_batch_paths):

            batch = self.load_tmp_batch(index)
            matrix = np.zeros((self.batch_size, len(self.repertoire_ids)))
            row_names = []

            for item_index, item in enumerate(batch):
                row_names.append(item)
                for repertoire_index, repertoire_id in enumerate(self.repertoire_ids):
                    if repertoire_id in batch[item]:
                        matrix[item_index][repertoire_index] = batch[item][repertoire_id]

            df = pd.DataFrame(matrix[:len(row_names)], index=row_names, columns=self.repertoire_ids)
            self.batch_paths.append(self.path + "batch_{}.csv".format(index))
            df.to_csv(self.batch_paths[-1])

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
        keep = []
        value = []
        item_to_update = []

        for item in items:
            if item in batch:
                value.append(1)
                item_to_update.append(item)
            else:
                keep.append(item)

        return keep, value, item_to_update

    def _remove_existing_items(self, new_items: list, batch: dict, batch_index: int, repertoire_id: str) -> list:

        update = {"value": [], "index": []}

        step = int(len(new_items) / self.pool_size) if int(len(new_items) / self.pool_size) > 0 else len(new_items)
        start, end = 0, step
        arguments = []
        while start < len(new_items):
            arguments.append((new_items[start:end], batch))
            end += step
            start += step

        with Pool(self.pool_size) as pool:
            output = pool.starmap(self._match_items_to_batch, arguments)

        flatten_and_remove_none = lambda l: [element for sublist in l for element in sublist if element is not None]

        new_items_to_keep = flatten_and_remove_none([output[i][0] for i in range(len(output))])
        update["value"] = flatten_and_remove_none([output[i][1] for i in range(len(output))])
        update["index"] = flatten_and_remove_none([output[i][2] for i in range(len(output))])

        for index, item in enumerate(update["index"]):
            batch[item][repertoire_id] = update["value"][index]

        self.store_tmp_batch(batch, batch_index)

        return new_items_to_keep

    def store_tmp_batch(self, batch: dict, batch_index: int):
        if batch_index == len(self.tmp_batch_paths):
            self.tmp_batch_paths.append(self.path + "batch_{}.pickle".format(batch_index))
        with open(self.tmp_batch_paths[batch_index], "wb") as file:
            pickle.dump(batch, file)

    def get_tmp_batches(self):
        for i in range(len(self.tmp_batch_paths)):
            yield self.load_tmp_batch(i)

    def load_tmp_batch(self, batch_index: int) -> dict:
        with open(self.tmp_batch_paths[batch_index], "rb") as file:
            batch = pickle.load(file)
        return batch

    def add_items_for_repertoire(self, items: list, repertoire_id: str):
        last_batch_index = len(self.tmp_batch_paths)-1
        batch = self.load_tmp_batch(last_batch_index)
        item_index = 0
        while len(batch) < self.batch_size and item_index < len(items):
            batch[items[item_index]] = {repertoire_id: 1}
            item_index += 1

        self.store_tmp_batch(batch, last_batch_index)

        self.item_count += len(items)

        items = items[item_index:]

        while len(items) > 0:
            batch = {item: {repertoire_id: 1} for item in items[:self.batch_size]}
            last_batch_index += 1
            self.store_tmp_batch(batch, last_batch_index)
            items = items[self.batch_size:]
