import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import shutil

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
# from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.ProGen import ProGen
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_progen():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'progen')

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.},
                                                               {}, path / 'dataset', region_type="IMGT_JUNCTION")

    progen = ProGen('beta',
                    'tokenizer.json',
                    'progen2-oas',
                    27,
                    1,
                    '1',
                    '2',
                    name='progen_test',
                    region_type="IMGT_JUNCTION",
                    seed=42)
    #
    # progen.fit(dataset, path / 'model')
    # progen.generate_sequences(7, 1, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    # assert (path / 'generated_dataset').exists()
    # assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()
    #
    # assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    shutil.rmtree(path)
