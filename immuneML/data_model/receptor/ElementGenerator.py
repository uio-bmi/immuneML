from itertools import chain
from pathlib import Path
from typing import List

import math
import numpy as np

from immuneML.data_model.bnp_util import bnp_read_from_file, bnp_write_to_file, make_element_dataset_objects, \
    merge_dataclass_objects
from immuneML.util.ReflectionHandler import ReflectionHandler


class ElementGenerator:

    def __init__(self, file_list: list, file_size: int = 10000, element_class_name: str = "", buffer_type=None):
        self.file_list = file_list
        self.file_lengths = [-1 for _ in range(len(file_list))]
        self.file_size = file_size
        self.element_class_name = element_class_name
        self.buffer_type = buffer_type

    def get_attribute(self, attribute: str):
        elements = []
        for file in self.file_list:
            bnp_data = bnp_read_from_file(file, self.buffer_type)
            elements.append(getattr(bnp_data, attribute))
        return np.concatenate(elements)

    def get_attributes(self, attributes: List[str]):
        elements = {attr: [] for attr in attributes}
        for file in self.file_list:
            bnp_data = bnp_read_from_file(file, self.buffer_type)
            for attribute in attributes:
                elements[attribute].append(getattr(bnp_data, attribute))
        return {attr: np.concatenate(elements[attr]) for attr in attributes}

    def _load_batch(self, current_file: int, return_objects: bool = True):

        element_class = ReflectionHandler.get_class_by_name(self.element_class_name, "data_model")
        assert hasattr(element_class, 'create_from_record'), \
            f"{ElementGenerator.__name__}: cannot load the binary file, the class {element_class.__name__} has no 'create_from_record' method."

        try:
            bnp_data = bnp_read_from_file(self.file_list[current_file], self.buffer_type)
            if return_objects:
                elements = make_element_dataset_objects(bnp_data, self.element_class_name)
            else:
                elements = bnp_data
        except ValueError as error:
            raise ValueError(f'{ElementGenerator.__name__}: an error occurred while creating an object from tsv file. '
                             f'Details: {error}')

        return elements

    def _get_element_count(self, file_index: int):

        if self.file_lengths[file_index] == -1:
            self.file_lengths[file_index] = len(bnp_read_from_file(self.file_list[file_index], self.buffer_type))

        return self.file_lengths[file_index]

    def get_element_count(self):
        for index in range(len(self.file_list)):
            if self.file_lengths[index] == -1:
                self._get_element_count(index)
        return sum(self.file_lengths)

    def build_batch_generator(self, return_objects: bool = True):
        """
        creates a generator which will return one batch of elements at the time

        :param batch_size: how many elements should be returned at once (default 1)
        :return: element generator
        """

        for current_file_index in range(len(self.file_list)):
            batch = self._load_batch(current_file_index, return_objects)
            yield batch

    def build_element_generator(self, return_objects: bool = True):
        """
        creates a generator which will return one element at the time
        :return: element generator
        """
        for current_file_index in range(len(self.file_list)):
            batch = self._load_batch(current_file_index, return_objects)
            if return_objects:
                for element in batch:
                    yield element
            else:
                yield batch

    def make_subset(self, example_indices: list, path: Path, dataset_type: str, dataset_identifier: str, paired: bool = False):
        if example_indices is None or len(example_indices) == 0:
            raise RuntimeError(f"{ElementGenerator.__name__}: no examples were specified to create the dataset subset. "
                               f"Dataset type was {dataset_type}, dataset identifier: {dataset_identifier}.")
        elements = None
        file_count = 1
        tmp_file_size = self.file_size if not paired else self.file_size * 2

        example_indices.sort()

        batch_filenames = self._prepare_batch_filenames(len(example_indices), path, dataset_type, dataset_identifier)

        for index, batch in enumerate(self.build_batch_generator(return_objects=False)):
            extracted_elements = self._extract_elements_from_batch(index, batch, example_indices, paired=paired)
            elements = merge_dataclass_objects([elements, extracted_elements]) if elements else extracted_elements

            if len(elements) >= tmp_file_size or len(elements) == len(example_indices):
                bnp_write_to_file(batch_filenames[file_count - 1], elements[:tmp_file_size])
                file_count += 1
                elements = elements[tmp_file_size:]

        if len(elements) > 0:
            bnp_write_to_file(batch_filenames[file_count - 1], elements)

        return batch_filenames

    def _prepare_batch_filenames(self, example_count: int, path: Path, dataset_type: str, dataset_identifier: str):
        batch_count = math.ceil(example_count / self.file_size)
        digits_count = len(str(batch_count)) + 1
        filenames = [
            path / f"{dataset_identifier}_{dataset_type}_batch{''.join(['0' for _ in range(digits_count - len(str(index)))])}{index}.tsv"
            for index in range(batch_count)]
        return filenames

    def _extract_elements_from_batch(self, index, batch, example_indices, paired: bool = False):
        if paired:
            upper_limit, lower_limit = (index + 1) * len(batch) / 2, index * len(batch) / 2
            batch_indices = list(chain.from_iterable([((ind - lower_limit) * 2, (ind - lower_limit) * 2 + 1)
                                                      for ind in example_indices if lower_limit <= ind < upper_limit]))
        else:
            upper_limit, lower_limit = (index + 1) * len(batch), index * len(batch)
            batch_indices = [ind - lower_limit for ind in example_indices if lower_limit <= ind < upper_limit]

        elements = batch[[int(i) for i in batch_indices]]
        return elements

    def get_data_from_index_range(self, start_index: int, end_index: int, obj_in_two_lines: bool = False):
        elements = []
        start_file_index = start_index // self.file_size
        end_file_index = min(len(self.file_list) - 1, end_index // self.file_size)

        for current_file_index in range(start_file_index, end_file_index + 1):
            batch = self._load_batch(current_file_index)
            i = start_index % self.file_size if current_file_index == start_file_index else 0
            while len(elements) < end_index - start_index + 1 and i < len(batch):
                elements.append(batch[i])
                i += 1

        return elements
