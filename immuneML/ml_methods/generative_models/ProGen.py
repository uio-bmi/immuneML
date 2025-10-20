import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, TrainingArguments, Trainer

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name, write_yaml, read_yaml
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.progen.ProGenConfig import ProGenConfig
from immuneML.ml_methods.generative_models.progen.ProGenForCausalLM import ProGenForCausalLM
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder


class ProGen(GenerativeModel):
    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        model_overview_file = path / 'model_overview.yaml'
        assert model_overview_file.exists(), f"{cls.__name__}: {model_overview_file} is not a file."

        # Uses ProGen weights/tokenizer paths from training. Override in model_overview.yaml if needed.
        model_overview = read_yaml(model_overview_file)
        progen = ProGen(**{k: v for k, v in model_overview.items() if k != 'type'})

        config = ProGenConfig.from_pretrained(path)
        model = ProGenForCausalLM.from_pretrained(path,
                                                  config=config,
                                                  dtype=torch.float32 if not progen.fp16 else torch.float16)

        progen.model = model.to(progen.device).eval()

        return progen

    def __init__(self, locus, tokenizer_path: Path, trained_model_path: Path, num_frozen_layers: int, num_epochs: int,
                 learning_rate: int, device: str, fp16: bool = False, prefix_text: str = "", suffix_text: str = "",
                 max_new_tokens: int = 1024, temperature: float = 1.0, top_p: float = 0.9, prompt: str = "1",
                 num_gen_batches: int = 1, per_device_train_batch_size: int = 2, remove_affixes: bool = True,
                 name: str = None, region_type: str = RegionType.IMGT_JUNCTION.name, seed: int = None, ):
        super().__init__(locus, seed=seed, name=name, region_type=RegionType.get_object(region_type))
        self.sequence_type = SequenceType.AMINO_ACID
        self.tokenizer_path = tokenizer_path
        self.trained_model_path = trained_model_path
        self.num_frozen_layers = num_frozen_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device  # "cpu" or "cuda"
        self.fp16 = fp16
        self.prefix_text = prefix_text
        self.suffix_text = suffix_text
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt = prompt
        self.num_gen_batches = num_gen_batches
        self.per_device_train_batch_size = per_device_train_batch_size
        self.remove_affixes = remove_affixes
        self.model = None

        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.hf_tokenizer.pad_token = "<|pad|>"
        self.hf_tokenizer.eos_token = "<|eos|>"
        self.hf_tokenizer.bos_token = "<|bos|>"

    def fit(self, data, path: Path = None):
        assert path is not None, "ProGen.fit requires a target directory path for training outputs."

        logs_dir, output_dir = self._prepare_training_paths(path)
        tokenized_dataset = self._preprocess_dataset(data)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.hf_tokenizer, mlm=False)
        config = ProGenConfig.from_pretrained(self.trained_model_path)
        model = ProGenForCausalLM.from_pretrained(self.trained_model_path, config=config)
        self._freeze_model_layers(model)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            use_cpu=True if self.device == "cpu" else False,
            save_safetensors=False,
            logging_dir=str(logs_dir),
            save_total_limit=1,
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.hf_tokenizer,
            data_collator=data_collator
        )

        print_log(f"{self.name or ProGen.__name__}: starting ProGen fine-tuning.", True)
        trainer.train()
        print_log(f"{self.name or ProGen.__name__}: finished ProGen fine-tuning.", True)
        self.model = trainer.model.to(self.device).eval()

    def _freeze_model_layers(self, model):
        for layer in model.transformer.h[:self.num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in model.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True

    def _preprocess_dataset(self, data):
        data_df = data.data.topandas()
        data_df["junction_aa"] = self.prefix_text + data_df["junction_aa"].astype(str).fillna("") + self.suffix_text
        hf_dataset = HFDataset.from_pandas(data_df[["junction_aa"]], preserve_index=False)
        tokenized_dataset = hf_dataset.map(
            self.hf_tokenizer,
            batched=True,
            input_columns="junction_aa",
            fn_kwargs={"truncation": True},
            remove_columns=["junction_aa"],
        )
        return tokenized_dataset

    def _prepare_training_paths(self, path):
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        output_dir = base_path / "model"
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = base_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir, output_dir

    def save_model(self, path: Path) -> Path:
        model_path = PathBuilder.build(path / 'model')
        self.model.save_pretrained(model_path, safe_serialization=False)
        shutil.copy(self.tokenizer_path, model_path / Path(self.tokenizer_path).name)

        skip_export_keys = {"model", "tokenizer", "hf_tokenizer", 'region_type', 'sequence_type'}
        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={**{k: v for k, v in vars(self).items() if k not in skip_export_keys},
                              **{'type': self.__class__.__name__,
                                 'locus': self.locus.name}})

        archive_path = path / f"trained_model_{self.name}.zip"
        with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
            for file_path in (fp for fp in model_path.rglob("*") if fp.is_file()):
                archive.write(file_path, file_path.relative_to(model_path))

        return archive_path.resolve()

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType,
                           compute_p_gen: bool) -> Dataset:
        prompt_encoding = self.hf_tokenizer(self.prompt, return_tensors="pt")
        prompt_input_ids = prompt_encoding.input_ids.to(self.device)
        prompt_attention_mask = prompt_encoding.attention_mask.to(self.device)
        gen_sequences = []

        num_sequences_per_batch = count // self.num_gen_batches
        for i in range(self.num_gen_batches):
            num_current_sequences = num_sequences_per_batch if i < self.num_gen_batches - 1 else count - (
                    num_sequences_per_batch * (self.num_gen_batches - 1))
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_return_sequences=num_current_sequences,
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    return_dict_in_generate=False
                )

            gen_sequences.extend(self.hf_tokenizer.batch_decode(output, skip_special_tokens=True))

            print_log(f"{self.name or ProGen.__name__}: {(i + 1) * num_current_sequences} sequences generated.", True)

        if self.remove_affixes:
            gen_sequences = self._remove_affixes(gen_sequences)

        gen_sequences_df = pd.DataFrame({get_sequence_field_name(self.region_type, self.sequence_type): gen_sequences,
                                         'locus': [self.locus.to_string() for _ in range(count)],
                                         'gen_model_name': [self.name for _ in range(count)]})

        return SequenceDataset.build_from_partial_df(gen_sequences_df, PathBuilder.build(path),
                                                     'synthetic_dataset', {'gen_model_name': [self.name]},
                                                     {'gen_model_name': str})

    def _remove_affixes(self, gen_sequences):
        prefix_text = self.hf_tokenizer.decode(self.hf_tokenizer(self.prefix_text).input_ids,
                                               skip_special_tokens=True)
        suffix_text = self.hf_tokenizer.decode(self.hf_tokenizer(self.suffix_text).input_ids,
                                               skip_special_tokens=True)
        gen_sequences = [seq.replace(prefix_text, '').replace(suffix_text, '') for seq in
                         gen_sequences]
        return gen_sequences

    def is_same(self, model) -> bool:
        raise NotImplementedError

    def compute_p_gens(self, sequences, sequence_type: SequenceType) -> np.ndarray:
        raise RuntimeError

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType) -> float:
        raise RuntimeError

    def can_compute_p_gens(self) -> bool:
        return False

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        return RuntimeError
