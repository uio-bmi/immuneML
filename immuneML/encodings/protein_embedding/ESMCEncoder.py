import numpy as np
import logging
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.Logger import log_memory_usage


class ESMCEncoder(ProteinEmbeddingEncoder):
    """
    Encoder based on a pretrained protein language model by Hayes et al. 2025. The used transformer model is
    "esmc_300m".

    Original publication:
    Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J.,
    Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A.,
    Mishra, C., Kim, C., … Rives, A. (2025). Simulating 500 million years of evolution with a language model.
    Science, 387(6736), 850–858. https://doi.org/10.1126/science.ads0018

    Original GitHub repository with license information: https://github.com/evolutionaryscale/esm

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
                my_emsc_encoder:
                    ESMC:
                        region_type: IMGT_CDR3
                        device: cpu
                        num_processes: 4
                        batch_size: 4096

    """

    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, device: str = 'cpu',
                 num_processes: int = 1, batch_size: int = 4096, scale_to_zero_mean: bool = True,
                 scale_to_unit_variance: bool = True):
        super().__init__(region_type, name, num_processes, device, batch_size, scale_to_zero_mean=scale_to_zero_mean,
                         scale_to_unit_variance=scale_to_unit_variance)
        self.transformer_link = 'esmc_300m'
        self.embedding_dim = 960

    @staticmethod
    def build_object(dataset: Dataset, **params):
        ParameterValidator.assert_region_type(params, ESMCEncoder.__name__)
        return ESMCEncoder(**{**params, 'region_type': RegionType[params['region_type'].upper()]})

    def _get_model(self, log_location):
        from esm.models.esmc import ESMC

        log_memory_usage(stage="start", location=log_location)
        logging.info(f"ESMC ({self.name}): Loading model: {self.transformer_link}")
        
        model = ESMC.from_pretrained(self.transformer_link)
        log_memory_usage("after model load", log_location)
        
        model = model.to(self.device).eval()
        log_memory_usage("after model to device", log_location)
        
        return model

    def _embed_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        import torch

        log_location = f"ESMCEncoder ({self.name})"
        model = self._get_model(log_location)

        sequences = getattr(sequence_set, seq_field)
        sequences = sequences.tolist()
        n_sequences = len(sequences)

        # Create memory-mapped array for embeddings
        embeddings = self._create_memmap_array((n_sequences, self.embedding_dim))

        # Process in batches
        for i in range(0, n_sequences, self.batch_size):
            batch_end = min(i + self.batch_size, n_sequences)
            batch = sequences[i:batch_end]
            
            logging.info(
                f"{log_location}: Processing batch {i//self.batch_size + 1}/{(n_sequences-1)//self.batch_size + 1}"
            )

            tokens = model._tokenize(batch)
            sequence_id = tokens != model.tokenizer.pad_token_id

            with torch.no_grad():
                output = model.forward(
                    sequence_tokens=tokens,
                    sequence_id=sequence_id
                )

            batch_embeddings = output.embeddings.cpu().numpy().mean(axis=1)
            embeddings[i:batch_end] = batch_embeddings

            del output, tokens, sequence_id
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_memory_usage(f"after batch {i//self.batch_size + 1}", log_location)

        logging.info(f"{log_location}: Finished processing all sequences")
        return embeddings

    def _get_encoding_name(self) -> str:
        return f"ESMC({self.transformer_link})"

    def _get_model_link(self) -> str:
        return self.transformer_link
