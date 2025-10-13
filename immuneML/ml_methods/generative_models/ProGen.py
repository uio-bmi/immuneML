from pathlib import Path

import numpy as np
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, TrainingArguments, Trainer

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.ml_methods.generative_models.progen.configuration_progen import ProGenConfig
from immuneML.ml_methods.generative_models.progen.modeling_progen import ProGenForCausalLM


class ProGen(GenerativeModel):
    @classmethod
    def load_model(cls, path: Path):
        pass

    def __init__(self, locus, tokenizer_path: Path, trained_model_path: Path, num_frozen_layers: int, num_epochs: int,
                 prefix_text: str = None, suffix_text: str = None, name: str = None,
                 region_type: str = RegionType.IMGT_JUNCTION.name, seed: int = None):
        super().__init__(locus, seed=seed, name=name, region_type=RegionType.get_object(region_type))
        self.prefix_text = prefix_text
        self.suffix_text = suffix_text
        self.tokenizer_path = tokenizer_path
        self.trained_model_path = trained_model_path
        self.num_frozen_layers = num_frozen_layers
        self.num_epochs = num_epochs

    def fit(self, data, path: Path = None):
        df = data.data.topandas()
        df["text"] = self.prefix_text + df["junction_aa"].astype(str).fillna("") + self.suffix_text
        hf_dataset = HFDataset.from_pandas(df[["text"]], preserve_index=False)

        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        hf_tokenizer.pad_token = "<|pad|>"

        def tokenize_function(examples):
            return hf_tokenizer(examples["text"], truncation=True)

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=hf_tokenizer,
            mlm=False
        )

        config = ProGenConfig.from_pretrained(self.trained_model_path)
        model = ProGenForCausalLM.from_pretrained(self.trained_model_path, config=config)

        for layer in model.transformer.h[:self.num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        for param in model.transformer.ln_f.parameters():
            param.requires_grad = True

        for param in model.lm_head.parameters():
            param.requires_grad = True

        training_args = TrainingArguments(
            output_dir=path,
            per_device_train_batch_size=2,
            num_train_epochs=self.num_epochs,
            learning_rate=5e-5,
            fp16=False,
            use_cpu=True,
            save_safetensors=False,
            logging_dir="./logs",
            save_total_limit=1,
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=hf_tokenizer,
            data_collator=data_collator
        )
        trainer.save_model(path)

    def save_model(self, path: Path) -> Path:
        pass

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType,
                           compute_p_gen: bool) -> Dataset:
        pass

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

