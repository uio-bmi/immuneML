import logging
import math
import shutil
from dataclasses import fields
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import List

import airr
import pandas as pd

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ElementDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.PathBuilder import PathBuilder


class AIRRExporter(DataExporter):
    """
    Exports a RepertoireDataset of Repertoires in AIRR format.

    Things to note:
        - one filename_prefix is given, which is combined with the Repertoire identifiers
        for the filenames, to create one file per Repertoire
        - 'counts' is written into the field 'duplicate_counts'
        - 'sequence_identifiers' is written both into the fields 'sequence_id' and 'rearrangement_id'

    """

    @staticmethod
    def export(dataset: Dataset, path: Path, number_of_processes: int = 1, omit_columns: list = None):
        PathBuilder.build(path)

        try:

            if isinstance(dataset, RepertoireDataset):
                repertoire_folder = "repertoires/"
                repertoire_path = PathBuilder.build(path / repertoire_folder)

                for repertoire in dataset.repertoires:
                    shutil.copyfile(repertoire.data_filename, repertoire_path / repertoire.data_filename.name)
                    shutil.copyfile(repertoire.metadata_filename, repertoire_path / repertoire.metadata_filename.name)

                shutil.copyfile(dataset.metadata_file, path / dataset.metadata_file.name)

            elif isinstance(dataset, ElementDataset):
                shutil.copyfile(dataset.filename, path / dataset.filename.name)
                shutil.copyfile(dataset.dataset_file, path / dataset.dataset_file.name)

        except shutil.SameFileError as e:
            logging.warning(f"AIRRExporter: target and input path are the same. Skipping the copy operation...")

        # TODO: add here export of full sequence if possible
