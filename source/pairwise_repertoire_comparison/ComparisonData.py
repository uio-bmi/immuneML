from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset


class ComparisonData:

    def __init__(self, repertoire_ids: list, matching_columns, item_columns: list, pool_size: int = 4, batch_size: int = 10000,
                 path: str = None):

        self.path = path
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.item_count = 0
        self.batch_paths = []
        self.update_paths = {}

        repertoire_columns = ["rep_{}".format(rep_id) for rep_id in repertoire_ids]
        repertoire_columns.extend(item_columns)
        self.item_columns = item_columns

        batch = pd.DataFrame(columns=repertoire_columns)
        self.batch_paths.append(self.path + "batch0.csv")
        batch.to_csv(self.batch_paths[-1], index=False)

        self.create_empty_update(0)

        self.matching_columns = matching_columns

    def get_repertoire_vector(self, identifier: str):
        repertoire_vector = np.zeros(self.item_count)
        for batch_index, batch in enumerate(self.get_batches(columns=["rep_{}".format(identifier)])):
            part = batch["rep_{}".format(identifier)].values
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
            self.process_repertoire(repertoire, repertoire.identifier, extract_items_fn)
        self.update_batches_from_log()

    def update_batch_from_log(self, batch_to_update):
        updates = pd.read_csv(self.update_paths[batch_to_update])
        batch = self.get_batch(self.batch_paths[batch_to_update])
        for index, update in updates.iterrows():
            batch[update["column"]][update["index"]] = update["value"]
        batch.to_csv(self.batch_paths[batch_to_update], index=False)

    def update_batches_from_log(self):
        with Pool(self.pool_size) as pool:
            pool.map(self.update_batch_from_log, [i for i in range(len(self.batch_paths))])

    def process_repertoire(self, repertoire, repertoire_id: str, extract_items_fn):
        items = pd.DataFrame(extract_items_fn(repertoire))
        items.drop_duplicates(inplace=True)
        new_items = self.filter_existing_items(items, repertoire_id)
        self.add_items_for_repertoire(new_items, repertoire_id)

    def filter_existing_items(self, items: pd.DataFrame, repertoire_id: str) -> pd.DataFrame:
        new_items = items
        for batch_index, batch in enumerate(self.get_batches()):
            new_items = self._remove_existing_items(new_items, batch, batch_index, repertoire_id)
        return new_items

    def _match_item_to_batch(self, item, batch, new_items_columns, index, repertoire_id):
        keep = None
        column = None
        value = None
        index_to_update = None
        matches = np.logical_and.reduce(item.values == batch[new_items_columns].values, axis=1)
        occurrences = np.sum(matches)
        if occurrences == 0:
            keep = index
        else:
            column = "rep_{}".format(repertoire_id)
            value = 1
            index_to_update = np.where(matches == True)[0][0]

        return keep, column, value, index_to_update

    def _remove_existing_items(self, new_items: pd.DataFrame, batch: pd.DataFrame, batch_index: int, repertoire_id: str) -> pd.DataFrame:

        new_items.reset_index(drop=True, inplace=True)

        update = {"column": [], "value": [], "index": []}

        arguments = [(item, batch, new_items.columns.values.tolist(), index, repertoire_id) for index, item in new_items.iterrows()]

        with Pool(4) as pool:
            output = pool.starmap(self._match_item_to_batch, arguments)

        indices_to_keep = [output[i][0] for i in range(new_items.shape[0]) if output[i][0] is not None]
        update["column"] = [output[i][1] for i in range(new_items.shape[0]) if output[i][1] is not None]
        update["value"] = [output[i][2] for i in range(new_items.shape[0]) if output[i][1] is not None]
        update["index"] = [output[i][3] for i in range(new_items.shape[0]) if output[i][1] is not None]

        update_path = self.path + "update_{}.csv".format(batch_index)
        self.update_paths[batch_index] = update_path
        pd.DataFrame(update).to_csv(update_path, index=False, mode="a", header=None)

        new_items = new_items.iloc[indices_to_keep]

        return new_items

    def add_items_for_repertoire(self, items: pd.DataFrame, repertoire_id: str):
        batch = self.get_batch(self.batch_paths[-1])

        columns_to_add = list(set(batch.columns.values.tolist()) - set(items.columns))
        for column in columns_to_add:
            items[column] = 0

        items.loc[:, "rep_{}".format(repertoire_id)] = 1
        batch = batch.append(items, ignore_index=True, sort=False)

        batch[:self.batch_size].to_csv(self.batch_paths[-1], index=False)
        batch = batch[self.batch_size:]
        while batch.shape[0] > 0:
            new_batch_path = self.path + "batch{}.csv".format(len(self.batch_paths))
            self.batch_paths.append(new_batch_path)
            batch[:self.batch_size].to_csv(new_batch_path, index=False)
            self.create_empty_update(len(self.batch_paths) - 1)
            batch = batch[self.batch_size:]

        self.item_count += len(items)

    def create_empty_update(self, batch_index: int):
        update_path = self.path + "update_{}.csv".format(batch_index)
        pd.DataFrame({"column": [], "value": [], "index": []}).to_csv(update_path, index=False)
        self.update_paths[batch_index] = update_path
