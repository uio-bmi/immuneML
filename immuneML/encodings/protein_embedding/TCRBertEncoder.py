import re
from itertools import zip_longest
import numpy as np
import logging
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.Logger import log_memory_usage


class TCRBertEncoder(ProteinEmbeddingEncoder):
    """
    TCRBertEncoder is based on `TCR-BERT <https://github.com/wukevin/tcr-bert/tree/main>`_, a large language model
    trained on TCR sequences. TCRBertEncoder embeds TCR sequences using either of the pre-trained models provided on
    HuggingFace repository.

    Original publication:
    Wu, K. E., Yost, K., Daniel, B., Belk, J., Xia, Y., Egawa, T., Satpathy, A., Chang, H., & Zou, J. (2024).
    TCR-BERT: Learning the grammar of T-cell receptors for flexible antigen-binding analyses. Proceedings of the
    18th Machine Learning in Computational Biology Meeting, 194–229. https://proceedings.mlr.press/v240/wu24b.html

    **Dataset type:**

    - SequenceDataset

    - ReceptorDataset

    - RepertoireDataset

    **Specification arguments:**

    - model (str): The pre-trained model to use (huggingface model hub identifier). Available options are 'tcr-bert'
      and 'tcr-bert-mlm-only'.

    - layers (list): The hidden layers to use for encoding. Layers should be given as negative integers, where -1
      indicates the last representation, -2 second to last, etc. Default is [-1].

    - method (str): The method to use for pooling the hidden states. Available options are 'mean', 'max'',
      'cls', and 'pool'. Default is 'mean'. For explanation of the methods, see GitHub repository of TCR-BERT.

    - batch_size (int): The number of sequences to encode at the same time. This could have large impact on memory usage.
      If memory is an issue, try with smaller batch sizes. Defaults to 4096.

    - scale_to_zero_mean (bool): Whether to scale the embeddings to zero mean. Defaults to True.

    - scale_to_unit_variance (bool): Whether to scale the embeddings to unit variance. Defaults to True.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_tcr_bert_encoder: TCRBert

    """
    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, model: str = None,
                 layers: list = None, method: str = None, batch_size: int = None, device: str = 'cpu',
                 scale_to_zero_mean: bool = True, scale_to_unit_variance: bool = True):
        super().__init__(region_type, name, num_processes=1, device=device, batch_size=batch_size,
                         scale_to_zero_mean=scale_to_zero_mean, scale_to_unit_variance=scale_to_unit_variance)
        self.model = model
        self.layers = layers
        self.method = method
        self.embedding_dim = 768

    @staticmethod
    def build_object(dataset: Dataset, **params):
        prepared_params = TCRBertEncoder._prepare_parameters(**params)
        return TCRBertEncoder(**prepared_params)

    @staticmethod
    def _prepare_parameters(name: str = None, model: str = None, layers: list = None, method: str = None,
                            batch_size: int = None, device: str = None):
        location = TCRBertEncoder.__name__
        ParameterValidator.assert_in_valid_list(model, ["tcr-bert", "tcr-bert-mlm-only"], location, "model")
        ParameterValidator.assert_type_and_value(layers, list, location, "layers")
        ParameterValidator.assert_in_valid_list(method, ["mean", "max", "attn_mean", "cls", "pool"], location, "method")
        ParameterValidator.assert_type_and_value(batch_size, int, location, "batch_size")
        ParameterValidator.assert_type_and_value(device, str, location, 'device')
        if len(re.findall("cuda:[0-9]*", device)) == 0:
            ParameterValidator.assert_in_valid_list(device, ['cpu', 'mps', 'cuda'], location, 'device')
        return {"name": name, "model": model, "layers": layers, "method": method, "batch_size": batch_size,
                'device': device}

    def _get_caching_params(self, dataset, params: EncoderParams, step: str = None):
        return (dataset.identifier, tuple(params.label_config.get_labels_by_name()), self.scale_to_zero_mean,
                self.scale_to_unit_variance, step, self.region_type.name, tuple(self.layers), self._get_encoding_name(),
                params.learn_model)

    def _get_model_and_tokenizer(self, log_location):
        from transformers import BertModel, BertTokenizer
        
        log_memory_usage(stage="start", location=log_location)
        logging.info(f"TCRBert ({self.name}): Loading model: wukevin/{self.model}")
        
        model = BertModel.from_pretrained(f"wukevin/{self.model}", attn_implementation="eager")
        log_memory_usage("after model load", log_location)
        
        model = model.to(self.device).eval()
        log_memory_usage("after model to device", log_location)
        
        tokenizer = BertTokenizer.from_pretrained(
            f"wukevin/{self.model}",
            do_basic_tokenize=False,
            do_lower_case=False,
            tokenize_chinese_chars=False,
            unk_token="?",
            sep_token="|",
            pad_token="$",
            cls_token="*",
            mask_token=".",
            padding_side="right"
        )
        log_memory_usage("after tokenizer load", log_location)
        
        return model, tokenizer

    def _embed_sequence_set(self, sequence_set, seq_field):
        import torch
        
        log_location = f"TCRBertEncoder ({self.name})"
        model, tokenizer = self._get_model_and_tokenizer(log_location)
        
        seqs = self._get_sequences(sequence_set, seq_field)
        n_sequences = len(seqs)
        
        # Calculate embedding dimension based on number of layers
        total_dim = len(self.layers) * self.embedding_dim if self.method != "pool" else self.embedding_dim
        
        # Create memory-mapped array for embeddings
        embeddings = self._create_memmap_array((n_sequences, total_dim))
        
        chunks = [seqs[i: i + self.batch_size] for i in range(0, n_sequences, self.batch_size)]
        chunks_pair = [None] * len(chunks)  # Create matching None pairs for zip_longest
        chunks_zipped = list(zip_longest(chunks, chunks_pair))
        
        current_idx = 0
        with torch.no_grad():
            for batch_idx, seq_chunk in enumerate(chunks_zipped):
                logging.info(
                    f"{log_location}: Processing batch {batch_idx + 1}/{len(chunks_zipped)}"
                )
                
                encoded = tokenizer(
                    *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                x = model.forward(**encoded, output_hidden_states=True, output_attentions=True)
                
                if self.method == "pool":
                    batch_size = len(seq_chunk[0])
                    embeddings[current_idx:current_idx + batch_size] = x.pooler_output.cpu().numpy()
                    current_idx += batch_size
                else:
                    batch_embeddings = []
                    for i in range(len(seq_chunk[0])):
                        e = []
                        for l in self.layers:
                            h = x.hidden_states[l][i].cpu().numpy()
                            
                            if self.method == "cls":
                                e.append(h[0])
                                continue
                                
                            if seq_chunk[1] is None:
                                seq_len = len(seq_chunk[0][i].split())
                            else:
                                seq_len = len(seq_chunk[0][i].split()) + len(seq_chunk[1][i].split()) + 1
                                
                            seq_hidden = h[1: 1 + seq_len]
                            
                            if self.method == "mean":
                                e.append(seq_hidden.mean(axis=0))
                            elif self.method == "max":
                                e.append(seq_hidden.max(axis=0))
                            else:
                                raise ValueError(f"Unrecognized method: {self.method}")
                                
                        e = np.hstack(e)
                        batch_embeddings.append(e)
                    
                    batch_embeddings = np.stack(batch_embeddings)
                    embeddings[current_idx:current_idx + len(batch_embeddings)] = batch_embeddings
                    current_idx += len(batch_embeddings)
                
                del x, encoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                log_memory_usage(f"after batch {batch_idx + 1}", log_location)
        
        logging.info(f"{log_location}: Finished processing all sequences")
        return embeddings

    def _get_sequences(self, sequence_set, field_name):
        seqs = getattr(sequence_set, field_name).tolist()
        seqs = [" ".join(list(s)) for s in seqs]
        return seqs

    def _get_encoding_name(self) -> str:
        return f"TCRBertEncoder({self.model})"

    def _get_model_link(self) -> str:
        return self.model
