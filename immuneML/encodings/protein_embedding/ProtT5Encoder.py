import numpy as np
import re
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator


class ProtT5Encoder(ProteinEmbeddingEncoder):

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
        super().__init__(region_type, name)
        self.region_type = region_type
        self.device = device
        self.transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        self.num_processes = num_processes

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ProtT5Encoder.__name__)
        return ProtT5Encoder(**params, region_type=RegionType[params['region_type'].upper()])

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

    def _get_encoding_name(self) -> str:
        return f"ProtT5Encoder({self.transformer_link})"

    def _get_model_link(self) -> str:
        return self.transformer_link

    def _get_caching_params(self, dataset, params: EncoderParams):
        cache_params = (dataset.identifier, self.__class__.__name__, self.region_type.name, self._get_model_link())
        return cache_params
