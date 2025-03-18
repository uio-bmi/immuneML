from itertools import zip_longest
import numpy as np
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder
from immuneML.util.ParameterValidator import ParameterValidator


class TCRBertEncoder(ProteinEmbeddingEncoder):
    """
    TCRBertEncoder is based on `TCR-BERT <https://github.com/wukevin/tcr-bert/tree/main>`, a large language model
    trained on TCR sequences. TCRBertEncoder embeds TCR sequences using either of the pre-trained models provided on
    HuggingFace repository.

    TCRBertEncoder should be used in combination with either dimensionality reduction or clustering methods.

        **Dataset type:**

        - SequenceDatasets

        **Specification arguments:**

        - model (str): The pre-trained model to use (huggingface model hub identifier). Available options are 'tcr-bert' and 'tcr-bert-mlm-only'.
        - layers (list): The hidden layers to use for encoding. Layers should be given as negative integers, where -1 indicates the last representation, -2 second to last, etc. Default is [-1].
        - method (str): The method to use for pooling the hidden states. Available options are 'mean', 'max', 'attn_mean',
            'cls', and 'pool'. Default is 'mean'. For explanation of the methods, see GitHub repository of TCR-BERT.
        - batch_size (int): The batch size to use for encoding. Default is 256.

        **YAML specification:**

        .. indent with spaces
        .. code-block:: yaml

            definitions:
                encodings:
                    my_tcr_bert_encoder: TCRBert

        """
    def __init__(self, name: str = None, region_type: RegionType = RegionType.IMGT_CDR3, model: str = None, layers: list = None, method: str = None,
                 batch_size: int = None):
        super().__init__(region_type, name)
        self.model = model
        self.layers = layers
        self.method = method
        self.batch_size = batch_size

    @staticmethod
    def build_object(dataset: Dataset, **params):
        prepared_params = TCRBertEncoder._prepare_parameters(**params)
        return TCRBertEncoder(**prepared_params)

    @staticmethod
    def _prepare_parameters(name: str = None, model: str = None, layers: list = None, method: str = None,
                            batch_size: int = None):
        location = TCRBertEncoder.__name__
        ParameterValidator.assert_in_valid_list(model, ["tcr-bert", "tcr-bert-mlm-only"], location, "model")
        ParameterValidator.assert_type_and_value(layers, list, location, "layers")
        ParameterValidator.assert_in_valid_list(method, ["mean", "max", "attn_mean", "cls", "pool"], location, "method")
        ParameterValidator.assert_type_and_value(batch_size, int, location, "batch_size")
        return {"name": name, "model": model, "layers": layers, "method": method, "batch_size": batch_size}

    def _get_caching_params(self, dataset, params: EncoderParams):
        return (("dataset_identifier", dataset.identifier),
                ("labels", tuple(params.label_config.get_labels_by_name())),
                ("encoding", TCRBertEncoder.__name__),
                ("learn_model", params.learn_model),
                ("step", ""),
                ("encoding_params", tuple(vars(self).items())))

    def _embed_sequence_set(self, sequence_set, seq_field):
        import torch
        seqs = self._get_sequences(sequence_set, seq_field)
        model, tok = self._get_relevant_model_and_tok()
        embeddings = []
        chunks_pair = [None]
        chunks = [seqs[i: i + self.batch_size] for i in range(0, len(seqs), self.batch_size)]
        chunks_zipped = list(zip_longest(chunks, chunks_pair))

        with torch.no_grad():
            for seq_chunk in chunks_zipped:
                encoded = tok(
                    *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
                )
                # manually calculated mask lengths
                # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
                input_mask = encoded["attention_mask"].numpy()
                encoded = {k: v.to('cpu') for k, v in encoded.items()}
                # encoded contains input attention mask of (batch, seq_len)
                x = model.forward(**encoded, output_hidden_states=True, output_attentions=True)
                if self.method == "pool":
                    embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                    continue

                for i in range(len(seq_chunk[0])):
                    e = []
                    for l in self.layers:
                        # Select the l-th hidden layer for the i-th example
                        h = (
                            x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                        )  # seq_len, hidden
                        # initial 'cls' token
                        if self.method == "cls":
                            e.append(h[0])
                            continue
                        # Consider rest of sequence
                        if seq_chunk[1] is None:
                            seq_len = len(seq_chunk[0][i].split())  # 'R K D E S' = 5
                        else:
                            seq_len = (
                                    len(seq_chunk[0][i].split())
                                    + len(seq_chunk[1][i].split())
                                    + 1  # For the sep token
                            )
                        seq_hidden = h[1: 1 + seq_len]  # seq_len * hidden
                        assert len(seq_hidden.shape) == 2
                        if self.method == "mean":
                            e.append(seq_hidden.mean(axis=0))
                        elif self.method == "max":
                            e.append(seq_hidden.max(axis=0))
                        elif self.method == "attn_mean":
                            # (attn_heads, seq_len, seq_len)
                            # columns past seq_len + 2 are all 0
                            # summation over last seq_len dim = 1 (as expected after softmax)
                            attn = x.attentions[l][i, :, :, : seq_len + 2]
                            # print(attn.shape)
                            print(attn.sum(axis=-1))
                            raise NotImplementedError
                        else:
                            raise ValueError(f"Unrecognized method: {self.method}")
                    e = np.hstack(e)
                    assert len(e.shape) == 1
                    embeddings.append(e)
        if len(embeddings[0].shape) == 1:
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.vstack(embeddings)
        del x
        del model
        torch.cuda.empty_cache()
        return embeddings

    def _get_sequences(self, sequence_set, field_name):
        seqs = getattr(sequence_set, field_name).tolist()
        seqs = [" ".join(list(s)) for s in seqs]
        return seqs

    def _get_relevant_model_and_tok(self):
        from transformers import BertModel, BertTokenizer
        model = BertModel.from_pretrained(f"wukevin/{self.model}")
        tok = BertTokenizer.from_pretrained(f"wukevin/{self.model}",
                                            do_basic_tokenize=False, do_lower_case=False,
                                            tokenize_chinese_chars=False, unk_token="?", sep_token="|",
                                            pad_token="$", cls_token="*", mask_token=".", padding_side="right")
        return model, tok

    def _get_encoding_name(self) -> str:
        return f"TCRBertEncoder({self.model})"

    def _get_model_link(self) -> str:
        return self.model
