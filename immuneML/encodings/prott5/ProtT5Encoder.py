import re
from multiprocessing import Pool

import numpy as np

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import RegionType, ChainPair, Chain
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.ParameterValidator import ParameterValidator


class ProtT5Encoder(DatasetEncoder):
    """
    Encoder based on a pretrained protein language model by Elnaggar et al. 2021. The used transformer model is
    "Rostlab/prot_t5_xl_half_uniref50-enc".

    Original publication:
    Elnaggar, A., Heinzinger, M., Dallago, C., Rihawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T.,
    Angerer, C., Steinegger, M., Bhowmik, D., & Rost, B. (2021). ProtTrans: Towards Cracking the Language of
    Life's Code Through Self-Supervised Deep Learning and High Performance Computing (No. arXiv:2007.06225).
    arXiv. https://doi.org/10.48550/arXiv.2007.06225

    Original GitHub repository with license information: https://github.com/agemagician/ProtTrans

    **Specification arguments:**

    - region_type (RegionType): Which part of the receptor sequence to encode. Defaults to IMGT_CDR3.

    - device (str): Which device to use for model inference - 'cpu', 'cuda', 'mps' - as defined by pytorch.
      Defaults to 'cpu'.

    - num_processes (int): Number of processes to use for parallel processing. Defaults to 1.

    """

    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, device: str = 'cpu',
                 num_processes: int = 1):
        super().__init__(name)
        self.region_type = region_type
        self.device = device
        self.transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        self.num_processes = num_processes

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ProtT5Encoder.__name__)
        return ProtT5Encoder(**params, region_type=RegionType[params['region_type'].upper()])

    def encode(self, dataset: Dataset, params: EncoderParams) -> Dataset:
        cache_params = (dataset.identifier, ProtT5Encoder.__name__, self.region_type.name, self.transformer_link)
        if isinstance(dataset, SequenceDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_sequence_dataset(dataset, params))
        elif isinstance(dataset, ReceptorDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_receptor_dataset(dataset, params))
        elif isinstance(dataset, RepertoireDataset):
            return CacheHandler.memo_by_params(cache_params, lambda: self._encode_repertoire_dataset(dataset, params))
        else:
            raise RuntimeError(f"{ProtT5Encoder.__name__}: {self.name}: invalid dataset type: {type(dataset)}.")

    def _encode_sequence_dataset(self, dataset: SequenceDataset, params: EncoderParams):

        seq_field = get_sequence_field_name(region_type=self.region_type, sequence_type=SequenceType.AMINO_ACID)
        transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        embeddings = self._embed_sequence_set(dataset.data, seq_field)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=np.array(embeddings),
                                                   labels={label.name: getattr(dataset.data, label.name).tolist()
                                                           for label in params.label_config.get_label_objects()},
                                                   example_ids=dataset.data.sequence_id.tolist(),
                                                   encoding=f'ProtT5Encoder({transformer_link})')

        return encoded_dataset

    def _encode_receptor_dataset(self, dataset: ReceptorDataset, params: EncoderParams):
        seq_field = get_sequence_field_name(region_type=self.region_type, sequence_type=SequenceType.AMINO_ACID)
        transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"

        data = dataset.data.topandas()
        loci = sorted(list(set(data.locus)))

        assert len(loci) == 2, (f"{ProtT5Encoder.__name__}: {self.name}: to encode receptor dataset, it has to include "
                                f"two different chains, but got: {loci} instead.")
        chain_pair = ChainPair.get_chain_pair([Chain.get_chain(locus) for locus in loci])

        chain1 = data[data.locus == loci[0]].sort_values('cell_id')
        chain2 = data[data.locus == loci[1]].sort_values('cell_id')

        assert all(chain1.cell_id.values == chain2.cell_id.values), (
            f"{ProtT5Encoder.__name__}: {self.name}: different number of "
            f"sequences per chain when sorted by cell_id:\n{chain1}\n{chain2}.")
        
        # Select rows from BNPDataClass using the sorted cell IDs
        chain1_data = dataset.data[dataset.data.locus.tolist() == loci[0]]
        chain2_data = dataset.data[dataset.data.locus.tolist() == loci[1]]

        # Get sorting indices
        chain1_order = np.argsort(chain1_data.cell_id)
        chain2_order = np.argsort(chain2_data.cell_id)

        # Apply sorting
        chain1_data = chain1_data[chain1_order]
        chain2_data = chain2_data[chain2_order]

        chain1_embeddings = self._embed_sequence_set(chain1_data, seq_field)
        chain2_embeddings = self._embed_sequence_set(chain2_data, seq_field)

        # Concatenate embeddings from both chains
        embeddings = np.concatenate((chain1_embeddings, chain2_embeddings), axis=1)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=embeddings,
                                                   labels={label.name: getattr(dataset, label.name) for label in
                                                           params.label_config.get_label_objects()},
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=f'ProtT5Encoder({transformer_link})')

        return encoded_dataset

    def _encode_repertoire_dataset(self, dataset: RepertoireDataset, params: EncoderParams):
        seq_field = get_sequence_field_name(region_type=self.region_type, sequence_type=SequenceType.AMINO_ACID)
        transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"

        examples = []
        for repertoire in dataset.repertoires:
            examples.append(CacheHandler.memo_by_params((repertoire.identifier, ProtT5Encoder.__name__,
                                                         self.region_type.name, self.transformer_link),
                                                        lambda: self._avg_sequence_set_embedding(
                                                            embedding=self._embed_sequence_set(repertoire.data,
                                                                                               seq_field))))

        encoded_dataset = dataset.clone()
        labels = dataset.get_metadata(params.label_config.get_labels_by_name())
        encoded_dataset.encoded_data = EncodedData(examples=np.array(examples), labels=labels,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=f'ProtT5Encoder({transformer_link})')

        return encoded_dataset

    def _avg_sequence_set_embedding(self, embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(axis=1)

    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str):
        import torch
        from transformers import T5Tokenizer, T5EncoderModel

        print("Loading: {}".format(self.transformer_link))
        model = T5EncoderModel.from_pretrained(self.transformer_link)
        if self.device == torch.device("cpu"):
            print("Casting model to full precision for running on CPU ...")
            model.to(torch.float32)  # only cast to full-precision if no GPU is available
        model = model.to(self.device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(self.transformer_link, do_lower_case=False, legacy=True)

        sequences = getattr(sequence_set, seq_field)
        sequence_lengths = sequences.lengths
        sequences = sequences.tolist()
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = [embedding_repr.last_hidden_state[i, :sequence_lengths[i]].mean(dim=0).numpy(force=True)
                      for i in range(len(sequences))]

        return np.array(embeddings)
