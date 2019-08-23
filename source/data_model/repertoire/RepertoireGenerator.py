# quality: gold

import pickle


class RepertoireGenerator:

    @staticmethod
    def load_repertoire(filename: str):
        with open(filename, "rb") as file:
            repertoire = pickle.load(file)
        return repertoire

    @staticmethod
    def load_batch(file_list):
        repertoires = []
        for filename in file_list:
            repertoire = RepertoireGenerator.load_repertoire(filename)
            repertoires.append(repertoire)
        return repertoires

    @staticmethod
    def build_item_generator(file_list: list, batch_size: int = 1):
        """
        creates a generator which will return one repertoire at the time;

        if repertoires are small and a few of them can fit into memory, it is
        possible to load a list of repertoires at once

        assumes that repertoires are stored in separate files

        :param file_list: list of file paths where repertoires can be found
        :param batch_size: how many repertoires should be loaded at once (default 1)
        :return: repertoire generator
        """
        index = 0
        file_count = len(file_list)
        batch_start, batch_end = -1, -1
        repertoires = []

        while index < file_count:
            if batch_start <= index < batch_end:
                repertoire = repertoires[index - batch_start]
            else:
                batch_start = index
                batch_end = index + batch_size
                repertoires = RepertoireGenerator.load_batch(file_list[batch_start:batch_end])
                repertoire = repertoires[0]

            index = index + 1

            yield repertoire

    @staticmethod
    def build_batch_generator(file_list: list, batch_size: int = 1):
        """
        creates a generator which will return a batch of repertoires at the time;

        assumes that repertoires are stored in separate files

        :param file_list: list of file paths where repertoires can be found
        :param batch_size: how many repertoires should be loaded at once (default 1)
        :return: repertoire generator
        """
        file_count = len(file_list)
        batch_start, batch_end = 0, batch_size

        while batch_start < file_count:
            repertoires = RepertoireGenerator.load_batch(file_list[batch_start:batch_end])

            batch_start = batch_end + 1
            batch_end = batch_start + batch_size

            yield repertoires
