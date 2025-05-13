import logging

import numpy as np

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.Logger import log_memory_usage
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

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets

    - RepertoireDatasets

    **Specification arguments:**

    - region_type (RegionType): Which part of the receptor sequence to encode. Defaults to IMGT_CDR3.

    - device (str): Which device to use for model inference - 'cpu', 'cuda', 'mps' - as defined by pytorch.
      Defaults to 'cpu'.

    - num_processes (int): Number of processes to use for parallel processing. Defaults to 1.

    - batch_size (int): The number of sequences to encode at the same time. This could have large impact on memory usage.
      If memory is an issue, try with smaller batch sizes. Defaults to 4096.

    - scale_to_zero_mean (bool): Whether to scale the embeddings to zero mean. Defaults to True.

    - scale_to_unit_variance (bool): Whether to scale the embeddings to unit variance. Defaults to True.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_prot_t5_encoder:
                    ProtT5::
                        region_type: IMGT_CDR3
                        device: cpu
                        num_processes: 1
                        batch_size: 4096

    """

    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, device: str = 'cpu',
                 num_processes: int = 1, batch_size: int = 4096, scale_to_zero_mean: bool = True,
                 scale_to_unit_variance: bool = True):
        super().__init__(region_type, name, num_processes, device, batch_size, scale_to_zero_mean=scale_to_zero_mean,
                         scale_to_unit_variance=scale_to_unit_variance)
        self.transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        self.batch_size = batch_size
        self.embedding_dim = 1024  # ProtT5's output dimension
        self.mem_map_path = None

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ProtT5Encoder.__name__)
        return ProtT5Encoder(**{**params, 'region_type': RegionType[params['region_type'].upper()]})

    def _get_model_and_tokenizer(self, log_location):
        import torch
        from transformers import T5Tokenizer, T5EncoderModel

        log_memory_usage(stage="start", location=log_location)
        logging.info(f"ProtT5 ({self.name}): Loading: {self.transformer_link}")
        model = T5EncoderModel.from_pretrained(self.transformer_link)
        log_memory_usage("after model load", log_location)

        if self.device == torch.device("cpu"):
            logging.info(f"{log_location}: Casting model to full precision for running on CPU ...")
            model.to(torch.float32)

        model = model.to(self.device).eval()
        log_memory_usage("after model to device", log_location)

        tokenizer = T5Tokenizer.from_pretrained(self.transformer_link, do_lower_case=False, legacy=True)
        log_memory_usage("after tokenizer load", log_location)

        return model, tokenizer

    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str):
        import torch

        log_location = f"ProtT5Encoder ({self.name})"
        model, tokenizer = self._get_model_and_tokenizer(log_location)

        sequences = getattr(sequence_set, seq_field)
        sequence_lengths = sequences.lengths
        sequences = [" ".join(list(sequence)) for sequence in sequences.tolist()]
        n_sequences = len(sequences)

        # Create memory-mapped array for embeddings
        embeddings = self._create_memmap_array((n_sequences, self.embedding_dim))

        for i in range(0, n_sequences, self.batch_size):
            batch_end = min(i + self.batch_size, n_sequences)
            batch = sequences[i:batch_end]
            batch_lengths = sequence_lengths[i:batch_end]

            logging.info(
                f"{log_location}: Processing batch {i // self.batch_size + 1}/{(n_sequences - 1) // self.batch_size + 1}")

            ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

            batch_embeddings = [embedding_repr.last_hidden_state[j, :batch_lengths[j]].mean(dim=0).cpu().numpy()
                                for j in range(len(batch))]
            embeddings[i:batch_end] = batch_embeddings

            del embedding_repr, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_memory_usage(f"after batch {i // self.batch_size + 1}", log_location)

        logging.info(f"{log_location}: Finished processing all sequences")
        return embeddings

    def _get_encoding_name(self) -> str:
        return f"ProtT5Encoder({self.transformer_link})"

    def _get_model_link(self) -> str:
        return self.transformer_link