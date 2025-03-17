
import torch
import sys
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import numpy as np
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.EncoderHelper import EncoderHelper
import os
import tempfile
from tqdm import tqdm


class Esm2Encoder(DatasetEncoder):
    def __init__(self, name: str = None):
        self.MODEL_LOCATION = "esm2_t33_650M_UR50D"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TOKS_PER_BATCH = 4096
        self.REPR_LAYERS = [-1]
        self.model, self.alphabet = pretrained.load_model_and_alphabet(self.MODEL_LOCATION)
        self.name = name
        self.context = None

    @staticmethod
    def build_object(dataset, **params):
        return Esm2Encoder(**params)

    def GetSequence(self, dataset):
        list_sequences = []
        for i in dataset.get_data():
            list_sequences.append(i.sequence_aa)
        print(f"Extracted sequences: {list_sequences}")
        return list_sequences

    def set_context(self, context: dict):
        self.context = context
        return self

    def createFasta(self, list_sequences):
        fasta_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        for i, seq in enumerate(list_sequences):
            fasta_file.write(f">{i}\n{seq}\n")
        fasta_file.close()
        print(f"Created temporary FASTA file: {fasta_file.name}")
        return fasta_file

    def _datasetcreation(self, fasta_file):
        dataset = FastaBatchedDataset.from_file(fasta_file.name)
        batches = dataset.get_batch_indices(self.TOKS_PER_BATCH, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self.alphabet.get_batch_converter(), batch_sampler=batches
        )
        os.remove(fasta_file.name)
        if os.path.exists(fasta_file.name):
            print(f"The file {fasta_file.name} is still here.")
        else:
            print(f"The file {fasta_file.name} has been used as input and then deleted.")
        print(f"Created DataLoader with {len(batches)} batches.")
        return data_loader, batches

    def get_embeddings(self, data_loader, batches):
        self.model.eval()
        self.model.to(self.device)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in self.REPR_LAYERS]
        mean_representations = []
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader, desc="Processing batches")):
                print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                for i, label in enumerate(labels):
                    mean_representation = [t[i, 1: len(strs[i]) + 1].mean(0).clone() for layer, t in representations.items()]
                    mean_representations.append(mean_representation[0])
        mean_representations = torch.vstack(mean_representations)
        mean_representations_np = mean_representations.numpy()
        return mean_representations_np

    def encode(self, dataset, params: EncoderParams):
        sequences = self.GetSequence(dataset)
        fasta_file = self.createFasta(sequences)
        data_loader, batches = self._datasetcreation(fasta_file)
        all_embeddings = self.get_embeddings(data_loader, batches)
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples= all_embeddings,
            encoding=Esm2Encoder.__name__,
            example_ids=None,
            labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )
        print("Encoded dataset created successfully.")

        return encoded_dataset