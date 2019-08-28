import math
import pickle


class ItemGenerator:

    def __init__(self, file_list: list, file_size: int = 1000):
        self.file_list = file_list
        self.file_lengths = [-1 for i in range(len(file_list))]
        self.file_size = file_size

    def _load_batch(self, cursor: dict, batch_size: int):
        items = []

        while len(items) < batch_size and cursor is not None:
            lines_to_read = batch_size - len(items)
            items.extend(self._load_from_file(cursor, lines_to_read))
            cursor = self._get_next_cursor(cursor, lines_to_read)

        return items, cursor

    def _load_from_file(self, cursor, lines_to_read):

        # TODO: make this abstract and move implementation to specific generator: load from csv not from pickle

        with open(self.file_list[cursor["file_index"]], "rb") as file:
            items = pickle.load(file)[cursor["line"]:cursor["line"] + lines_to_read]
        return items

    def _get_item_count(self, file_index: int):

        # TODO: make this abstract and move implementation to specific generator: count items in file for new format

        if self.file_lengths[file_index] == -1:
            with open(self.file_list[file_index], "rb") as file:
                count = len(pickle.load(file))
            self.file_lengths[file_index] = count

        return self.file_lengths[file_index]

    def _get_next_cursor(self, cursor, lines_to_read):
        lines = self._get_item_count(cursor["file_index"])

        if lines > cursor["line"] + lines_to_read:
            return {
                "file_index": cursor["file_index"],
                "line": cursor["line"] + lines_to_read
            }
        elif self._has_more_files(cursor):
            return {
                "file_index": cursor["file_index"] + 1,
                "line": 0
            }
        else:
            return None

    def _has_more_files(self, cursor: dict):
        return cursor["file_index"] != len(self.file_list) - 1

    def get_item_count(self):
        for index in range(len(self.file_list)):
            if self.file_lengths[index] == -1:
                self._get_item_count(index)
        return sum(self.file_lengths)

    def build_batch_generator(self, batch_size: int):
        """
        creates a generator which will return one batch of items at the time

        :param batch_size: how many items should be returned at once (default 1)
        :return: item generator
        """
        cursor = {
            "file_index": 0,
            "line": 0
        }

        while cursor is not None:
            batch, cursor = self._load_batch(cursor, batch_size)
            yield batch

    def build_item_generator(self, batch_size: int):
        """
        creates a generator which will return one item at the time

        :param batch_size: how many items should be loaded at once (default 1)
        :return: item generator
        """
        cursor = {
            "file_index": 0,
            "line": 0
        }

        while cursor is not None:
            batch, cursor = self._load_batch(cursor, batch_size)
            for item in batch:
                yield item

    def make_subset(self, example_indices: list, path: str):
        batch_size = 1000
        items = []
        file_count = 1

        example_indices.sort()

        batch_filenames = self._prepare_batch_filenames(len(example_indices), path)

        for index, batch in enumerate(self.build_batch_generator(batch_size)):
            extracted_items = self._extract_items_from_batch(index, batch_size, batch, example_indices)
            items.extend(extracted_items)

            if len(items) >= self.file_size or len(items) == len(example_indices):
                self._store_items_to_file(batch_filenames[file_count-1], items[:self.file_size])
                file_count += 1
                items = items[self.file_size:]

        return batch_filenames

    def _prepare_batch_filenames(self, example_count: int, path: str):
        batch_count = math.ceil(example_count / self.file_size)
        digits_count = len(str(batch_count))
        filenames = [path + "batch".join(["0" for i in range(digits_count-len(str(index)))]) + str(index) + ".pkl"
                     for index in range(digits_count)]
        return filenames

    def _store_items_to_file(self, path, items):
        with open(path, "wb") as file:
            pickle.dump(items, file)

    def _extract_items_from_batch(self, index, batch_size, batch, example_indices):
        upper_limit, lower_limit = (index + 1) * batch_size, index * batch_size
        batch_indices = [ind for ind in example_indices if lower_limit <= ind <= upper_limit]
        items = [batch[i - lower_limit] for i in batch_indices]
        return items
