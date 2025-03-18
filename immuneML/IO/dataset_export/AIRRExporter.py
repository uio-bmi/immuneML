import logging
import shutil
from pathlib import Path

import bionumpy
import pandas as pd

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.bnp_util import read_yaml, write_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ElementDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.util.PathBuilder import PathBuilder


class AIRRExporter(DataExporter):
    """
    Exports a RepertoireDataset of Repertoires in AIRR format.

    Things to note:

    - one filename_prefix is given, which is combined with the Repertoire identifiers for the filenames, to create one file per Repertoire
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
                    AIRRExporter.process_and_store_data_file(repertoire.data_filename,
                                                             repertoire_path / repertoire.data_filename.name)
                    shutil.copyfile(repertoire.metadata_filename, repertoire_path / repertoire.metadata_filename.name)

                shutil.copyfile(dataset.metadata_file, path / dataset.metadata_file.name)
                if dataset.dataset_file and dataset.dataset_file.is_file():
                    shutil.copyfile(dataset.dataset_file, path / dataset.dataset_file.name)

            elif isinstance(dataset, ElementDataset):
                AIRRExporter.process_and_store_data_file(dataset.filename, path / dataset.filename.name)
                AIRRExporter.process_and_store_dataset_file(dataset.dataset_file, path / dataset.dataset_file.name)

        except shutil.SameFileError as e:
            logging.warning(f"AIRRExporter: target and input path are the same. Skipping the copy operation...")

        # TODO: add here export of full sequence if possible

    @staticmethod
    def process_and_store_dataset_file(input_filename, output_filename):
        metadata = read_yaml(input_filename)
        metadata['filename'] = str(Path(metadata['filename']).name)
        write_yaml(output_filename, metadata)

    @staticmethod
    def process_and_store_data_file(input_filename, output_filename):
        df = pd.read_csv(input_filename, sep='\t', keep_default_na=False,
                         dtype={key: key_type if not isinstance(key_type, bionumpy.encodings.Encoding) else str
                                for key, key_type in AIRRSequenceSet.get_field_type_dict().items()})
        df = AIRRExporter.add_cdr3_from_junction(df)
        df.to_csv(output_filename, sep='\t', index=False)

    @staticmethod
    def add_cdr3_from_junction(df: pd.DataFrame) -> pd.DataFrame:
        if 'junction' in df.columns and 'cdr3' in df.columns and any(df.cdr3.eq('')):
            missing_cdr3 = df.cdr3.eq('')
            df.loc[missing_cdr3, 'cdr3'] = df.loc[missing_cdr3, 'junction'].str[3:-3]
        if 'junction_aa' in df.columns and 'cdr3_aa' in df.columns and any(df.cdr3_aa.eq('')):
            missing_cdr3_aa = df.cdr3_aa.eq('')
            df.loc[missing_cdr3_aa, 'cdr3_aa'] = df.loc[missing_cdr3_aa, 'junction_aa'].str[1:-1]
        return df
