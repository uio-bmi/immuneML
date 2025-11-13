import os
import shutil
import urllib
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizerFast

from immuneML.data_model.SequenceParams import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.ProGen import ProGen
from immuneML.ml_methods.generative_models.progen.ProGenConfig import ProGenConfig
from immuneML.ml_methods.generative_models.progen.ProGenForCausalLM import ProGenForCausalLM
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"Saved to {dest}")
    return dest


def make_dummy_progen_model(model_path: Path) -> Path:
    model_path.mkdir(parents=True, exist_ok=True)

    tokenizer_json = model_path / "tokenizer.json"
    download_file(
        "https://raw.githubusercontent.com/enijkamp/progen2/main/tokenizer.json",
        tokenizer_json
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
    tokenizer.pad_token = tokenizer.eos_token

    config = ProGenConfig(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_ctx=256,
        n_embd=16,
        n_layer=1,
        n_head=8,
    )

    model = ProGenForCausalLM(config)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    return model_path


def test_progen():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'progen')
    os.makedirs(path, exist_ok=True)

    model_dir = make_dummy_progen_model(path / "dummy_model")
    tokenizer_json = model_dir / "tokenizer.json"

    dataset = RandomDatasetGenerator.generate_sequence_dataset(20, {10: 1.},
                                                               {}, path / 'dataset', region_type="IMGT_JUNCTION")

    progen = ProGen(Chain.get_chain('beta'),
                    tokenizer_json,
                    model_dir,
                    1,
                    1,
                    5e-5,
                    'cpu',
                    prefix_text='',
                    suffix_text='',
                    name='progen_test',
                    region_type="IMGT_JUNCTION",
                    seed=42)

    progen.fit(dataset, path / 'model')
    progen.generate_sequences(7, 42, path / 'generated_dataset', SequenceType.AMINO_ACID, False)

    assert (path / 'generated_dataset').exists()
    assert (path / 'generated_dataset/synthetic_dataset.tsv').exists()
    assert pd.read_csv(str(path / 'generated_dataset/synthetic_dataset.tsv'), sep='\t').shape[0] == 7

    shutil.rmtree(path)
