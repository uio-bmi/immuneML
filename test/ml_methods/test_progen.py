import os
import shutil
import tarfile
import urllib
from pathlib import Path

import pandas as pd
import pytest

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.ProGen import ProGen
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"Saved to {dest}")
    return dest


@pytest.mark.skip(reason="Disabled by default to avoid large downloads during testing. Enable manually when needed.")
def test_progen():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'progen')
    os.makedirs(path, exist_ok=True)

    # Download ProGen2 model and tokenizer
    tokenizer_json = path / "tokenizer.json"
    model_tgz = path / "progen2-oas.tar.gz"
    model_dir = path / "progen2-oas"
    download_file("https://raw.githubusercontent.com/enijkamp/progen2/main/tokenizer.json", tokenizer_json)
    download_file("https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-oas.tar.gz", model_tgz)
    with tarfile.open(model_tgz, "r:gz") as tf:
        tf.extractall(model_dir)

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.},
                                                               {}, path / 'dataset', region_type="IMGT_JUNCTION")

    progen = ProGen('beta',
                    str(tokenizer_json),
                    str(model_dir),
                    27,
                    1,
                    5e-5,
                    'cpu',
                    prefix_text='1',
                    suffix_text='2',
                    name='progen_test',
                    region_type="IMGT_JUNCTION",
                    seed=42)
    progen.fit(dataset, path / 'model')
    progen.generate_sequences(7, 42, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()
    assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    shutil.rmtree(path)
