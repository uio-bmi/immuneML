
import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.EncoderHelper import EncoderHelper
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Antiberta2Encoder(DatasetEncoder):
    def __init__(self, name: str = None):
        self.batch_size = 2048
        self.model_name = "alchemab/antiberta2"
        self.tokenizer = RoFormerTokenizer.from_pretrained(self.model_name)
        self.name = name
        self.context = None

    @staticmethod
    def build_object(dataset, **params):
        return Antiberta2Encoder(**params)

    def _GetSequence(self, dataset):
        list_sequences = []
        for i in dataset.get_data():
            list_sequences.append(i.sequence_aa)
        return list_sequences

    def _split_characters(self, list_sequences):
        split_seq = []
        for x in list_sequences:
            split_seq.append(" ".join(x))
        return split_seq

    def _sequencetokenizer(self, split_seq):
        input_ids = []
        attention_masks = []
        for seq in split_seq:
            tokens = self.tokenizer(seq, truncation=True, padding='max_length', return_tensors="pt",
                                   add_special_tokens=True, max_length=200)
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])
        return input_ids, attention_masks

    def _datasetcreation(self, input_ids, attention_masks):
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        return data_loader

    def _embeddigstorage(self, data_loader):
        all_embeddings = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = RoFormerModel.from_pretrained(self.model_name).to(device)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                first_token_batch = embeddings[:, 0, :]
                all_embeddings.append(first_token_batch.cpu())
            all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy()

    def set_context(self, context: dict):
        self.context = context
        return self

    def encode(self, dataset, params: EncoderParams):
        # Get sequences from the dataset
        print("Starting encoding process...")
        sequences = self._GetSequence(dataset)
        #print(f"Extracted {len(sequences)} sequences from the dataset.")

        # Tokenize and encode the sequences
        input_ids, attention_masks = self._sequencetokenizer(self._split_characters(sequences))
        data_loader = self._datasetcreation(input_ids, attention_masks)
        all_embeddings = self._embeddigstorage(data_loader)

        # Create the encoded dataset
        print("Creating encoded dataset...")
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=all_embeddings,
            encoding=Antiberta2Encoder.__name__,
            example_ids=None,  # No example IDs
            labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )
        print("Encoded dataset created successfully.")
        return encoded_dataset