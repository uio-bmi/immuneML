import numpy as np
import pandas as pd


class ComparisonData:

    def __init__(self, repertoire_count: int, matching_columns, item_columns: list, path: str = None, batch_size: int = 10000):

        self.path = path
        self.batch_size = batch_size
        self.item_count = 0
        self.batch_paths = []

        repertoire_columns = ["rep_{}".format(ind+1) for ind in range(repertoire_count)]
        repertoire_columns.extend(item_columns)
        self.item_columns = item_columns

        batch = pd.DataFrame(columns=repertoire_columns)
        self.batch_paths.append(self.path + "batch1.csv")
        batch.to_csv(self.batch_paths[-1], index=False)

        self.matching_columns = matching_columns

    def get_repertoire_vector(self, index: int):
        repertoire_vector = np.zeros(self.item_count)
        for batch_index, batch in enumerate(self.get_batches(columns=["rep_{}".format(index)])):
            part = batch["rep_{}".format(index)].values
            start = batch_index * self.batch_size
            end = start + part.shape[0]
            repertoire_vector[start: end] = part
        return repertoire_vector

    def get_item_vector(self, index: int):
        batch_index = int(index / self.batch_size)
        index_in_batch = index - (batch_index * self.batch_size)
        return pd.read_csv(self.batch_paths[batch_index], nrows=1, skiprows=index_in_batch-1).iloc[0].values

    def get_batches(self, columns: list = None):
        for batch_path in self.batch_paths:
            yield self.get_batch(batch_path, columns)

    def get_batch(self, batch_path: str, columns: list = None):
        if columns is not None:
            return pd.read_csv(batch_path, usecols=columns)
        else:
            return pd.read_csv(batch_path)

    def process_repertoire(self, repertoire, repertoire_index: int, extract_items_fn):
        items = pd.DataFrame(extract_items_fn(repertoire))
        new_items = self.filter_existing_items(items, repertoire_index)
        self.add_items_for_repertoire(new_items, repertoire_index)

    def filter_existing_items(self, items: pd.DataFrame, repertoire_index: int) -> pd.DataFrame:
        new_items = items
        for batch_index, batch in enumerate(self.get_batches()):
            new_items = self._remove_existing_items(new_items, batch, self.batch_paths[batch_index], repertoire_index)
        return new_items

    def _remove_existing_items(self, new_items: pd.DataFrame, batch: pd.DataFrame, batch_path: str, repertoire_index: int) -> pd.DataFrame:

        # TODO: set existing vector in batch to 1 if an item exists in repertoire!!!

        swap_count = 0
        max_swaps = new_items.shape[0]
        while swap_count < max_swaps:
            size = min(new_items.shape[0], batch.shape[0])
            indices_to_keep = np.logical_and.reduce(new_items.values[:size] != batch[new_items.columns.tolist()].values[:size], axis=1)
            batch[:size]["rep_{}".format(repertoire_index)] = np.logical_not(indices_to_keep).astype(int)
            batch.to_csv(batch_path, index=False)
            new_items[:size] = new_items[:size][indices_to_keep]
            new_items = new_items.dropna()
            new_items = self._rotate_dataframe(new_items)
            swap_count += 1
        return new_items

    def _rotate_dataframe(self, df: pd.DataFrame):
        tmp_1 = df.iloc[-1].copy()
        df = df.shift(periods=1)
        df.iloc[0] = tmp_1
        return df

    def add_items_for_repertoire(self, items: pd.DataFrame, repertoire_index: int):
        batch = self.get_batch(self.batch_paths[-1])

        columns_to_add = list(set(batch.columns.values.tolist()) - set(items.columns))
        for column in columns_to_add:
            items[column] = 0

        items["rep_{}".format(repertoire_index)] = 1
        batch = batch.append(items, ignore_index=True)

        batch[:self.batch_size].to_csv(self.batch_paths[-1], index=False)
        batch = batch[self.batch_size:]
        while batch.shape[0] > 0:
            new_batch_path = self.path + "batch{}.csv".format(len(self.batch_paths) + 1)
            self.batch_paths.append(new_batch_path)
            batch[:self.batch_size].to_csv(new_batch_path, index=False)
            batch = batch[self.batch_size:]

        self.item_count += len(items)
