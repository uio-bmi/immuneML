import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import write_yaml, read_yaml, get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class TCRpeg(GenerativeModel):
    """
    TCRpeg is a deep autoregressive generative model of T-cell receptor CDR3 amino acid sequences. The probability of a
    sequence x is factorized as ``p(x) = p(x_1) * prod_i p(x_i | x_1, ..., x_(i-1)) * p(end | x)``, where the
    conditional probabilities are given by stacked GRU layers followed by a softmax. The amino acid input embeddings
    are word2vec embeddings and are kept fixed while the GRU layers are trained with the negative log-likelihood loss.
    New sequences are sampled residue by residue from the start token until the end token is sampled. The model gives
    the probability of any sequence, so generation probabilities (p_gen) can be computed.

    Optionally (``vj: True``), two single-layer fully connected heads predict the V and J gene from the final GRU
    hidden states, so that ``p(x, V, J) = p(x) * p(V | x) * p(J | x)``, and the generated sequences also include
    ``v_call`` and ``j_call``. The V and J genes are taken from the training data exactly as they are written there.

    This class is a wrapper around the ``tcrpeg`` Python package (https://github.com/jiangdada1221/TCRpeg, PyPI package
    tcrpeg, released under the GPLv3 licence), which has to be installed separately (e.g., by installing immuneML
    with the ``gen_models`` extra). The TCRpeg-c classifier from the same publication is not included.

    By default, the pretrained amino acid embeddings shipped with the package (tcrpeg/data/embedding_32.txt) are used.
    They were learned by the authors with word2vec on a large pool of human TRB CDR3 sequences (Emerson et al.). For
    other loci or species, the embeddings can instead be learned on the training data (``train_aa_embeddings: True``,
    using the package's word2vec implementation) or read from a user-provided file (``aa_embedding_path``). Note that
    the package's word2vec starts from random embeddings, uses plain gradient descent and already reduces its learning
    rate to 20% after the first epoch, so with the default word2vec parameters (learning rate 0.0001, 20 epochs) the
    learned embeddings stay very close to their random initialization. Since the embeddings are kept fixed while
    training the GRU layers, it is recommended to either provide pretrained embeddings through ``aa_embedding_path`` or
    to considerably increase word2vec_learning_rate and/or word2vec_epochs when using ``train_aa_embeddings``.

    The default architecture and training parameters are those of the package's training script
    (tcrpeg/scripts/train.py): 3 GRU layers with hidden size 64, batch size 1000, learning rate 0.0001, 20 epochs. The
    package reduces the learning rate to 20% of its value halfway through training. These defaults were chosen for
    very large repertoires (the publication trains on about 10^8 sequences). For small datasets (thousands to tens of
    thousands of sequences), such settings result in very few optimization steps and an undertrained model; a higher
    learning rate (e.g., 0.001, as in the package README example) together with a smaller batch size and/or more
    epochs is recommended. The package also uses only full batches: in each epoch, the training data is shuffled and
    the remaining sequences that do not fill a complete batch are skipped (e.g., with 1500 sequences and batch size
    1000, only 1000 sequences are used per epoch). Choosing a batch size that is much smaller than the number of
    training sequences, or divides it, limits this effect.

    Before training, sequences with non-standard amino acids, longer than max_length, or explicitly marked as
    non-productive are removed, as in the publication. If the region type is IMGT_JUNCTION, sequences that do not start
    with a cysteine are removed as well (unless require_c_start is False). If vj is True, sequences without V or J
    gene are removed.

    Original publication:

    Jiang, Y., Li, S. C. (2023). Deep autoregressive generative models capture the intrinsics embedded in T-cell
    receptor repertoires. Briefings in Bioinformatics, 24(2), bbad038. https://doi.org/10.1093/bib/bbad038

    **Specification arguments:**

    - locus (str): the locus of the receptor chain; if not set, it is taken from the training dataset

    - region_type (str): which part of the sequence to model, e.g., IMGT_JUNCTION (default, CDR3 including the
      conserved C and F/W, as in the publication) or IMGT_CDR3

    - hidden_size (int): number of features in the hidden state of each GRU layer; default 64

    - num_layers (int): number of stacked GRU layers; default 3

    - dropout (float): dropout between the GRU layers; default 0

    - num_epochs (int): number of training epochs; default 20

    - batch_size (int): number of sequences per batch for training, generation and computing p_gen; if there are fewer
      training sequences than batch_size, all of them are used as one batch; otherwise, sequences that do not fill a
      complete batch are skipped in each epoch; default 1000

    - learning_rate (float): learning rate for the Adam optimizer; default 0.0001

    - max_length (int): maximum sequence length; longer training sequences are removed, generated sequences are at
      most this long, and longer sequences get p_gen 0; default 30

    - vj (bool): whether to also model V and J gene usage conditioned on the sequence and output v_call and j_call for
      generated sequences; default False

    - require_c_start (bool): if True and region_type is IMGT_JUNCTION, training sequences that do not start with C are
      removed; default True

    - aa_embedding_path (str): optional path to a comma-separated file with amino acid embeddings in the TCRpeg format
      (22 rows: the token followed by the embedding values, with tokens 's' for start, the 20 amino acids and 'e' for
      end); if not set and train_aa_embeddings is False, the pretrained embeddings shipped with the package are used

    - train_aa_embeddings (bool): if True, amino acid embeddings are learned on the training data with the package's
      word2vec implementation instead of using the pretrained ones; with the default word2vec parameters, the learned
      embeddings stay close to random (see above); default False

    - embed_size (int): dimension of the embeddings learned when train_aa_embeddings is True (otherwise the dimension
      is given by the embedding file); default 32

    - word2vec_epochs (int): number of word2vec epochs when train_aa_embeddings is True; default 20

    - word2vec_batch_size (int): word2vec batch size (number of amino acid pairs); default 1000

    - word2vec_learning_rate (float): word2vec learning rate; default 0.0001

    - word2vec_window_size (int): word2vec context window size; default 2

    - device (str): torch device to use, e.g., cpu or cuda:0; default cpu

    - seed (int): random seed for the model or None


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                my_tcrpeg:
                    TCRpeg:
                        region_type: IMGT_JUNCTION
                        hidden_size: 64
                        num_layers: 3
                        num_epochs: 20
                        batch_size: 1000
                        learning_rate: 0.0001
                        max_length: 30
                        vj: False
                        device: cpu
                        seed: 1

    """

    EMBEDDING_FILE_NAME = 'aa_embeddings.txt'
    STATE_DICT_FILE_NAME = 'state_dict.pt'

    @classmethod
    def load_model(cls, path: Path):
        assert path.exists(), f"{cls.__name__}: {path} does not exist."

        for file in [path / 'model_overview.yaml', path / cls.STATE_DICT_FILE_NAME, path / cls.EMBEDDING_FILE_NAME]:
            assert file.exists(), f"{cls.__name__}: {file} is not a file."

        model_overview = read_yaml(path / 'model_overview.yaml')
        genes = model_overview.pop('genes', {'v': [], 'j': []})
        model = TCRpeg(**{k: v for k, v in model_overview.items() if k != 'type'})
        model._v_genes, model._j_genes = list(genes['v']), list(genes['j'])
        model._embedding_file = path / cls.EMBEDDING_FILE_NAME
        model._tcrpeg = model._make_tcrpeg(state_dict_file=path / cls.STATE_DICT_FILE_NAME)
        return model

    def __init__(self, locus: str = None, region_type: str = RegionType.IMGT_JUNCTION.name, hidden_size: int = 64,
                 num_layers: int = 3, dropout: float = 0., num_epochs: int = 20, batch_size: int = 1000,
                 learning_rate: float = 0.0001, max_length: int = 30, vj: bool = False, require_c_start: bool = True,
                 aa_embedding_path: str = None, train_aa_embeddings: bool = False, embed_size: int = 32,
                 word2vec_epochs: int = 20, word2vec_batch_size: int = 1000, word2vec_learning_rate: float = 0.0001,
                 word2vec_window_size: int = 2, device: str = 'cpu', name: str = None, seed: int = None):
        super().__init__(locus, name=name, region_type=RegionType.get_object(region_type), seed=seed)
        location = TCRpeg.__name__

        for param_name, value in [('hidden_size', hidden_size), ('num_layers', num_layers), ('num_epochs', num_epochs),
                                  ('batch_size', batch_size), ('max_length', max_length), ('embed_size', embed_size),
                                  ('word2vec_epochs', word2vec_epochs), ('word2vec_batch_size', word2vec_batch_size),
                                  ('word2vec_window_size', word2vec_window_size)]:
            ParameterValidator.assert_type_and_value(value, int, location, param_name, min_inclusive=1)
        for param_name, value in [('vj', vj), ('require_c_start', require_c_start),
                                  ('train_aa_embeddings', train_aa_embeddings)]:
            ParameterValidator.assert_type_and_value(value, bool, location, param_name)
        ParameterValidator.assert_type_and_value(dropout, (int, float), location, 'dropout', min_inclusive=0,
                                                 max_exclusive=1)
        for param_name, value in [('learning_rate', learning_rate), ('word2vec_learning_rate', word2vec_learning_rate)]:
            ParameterValidator.assert_type_and_value(value, (int, float), location, param_name, min_exclusive=0)
        ParameterValidator.assert_type_and_value(aa_embedding_path, str, location, 'aa_embedding_path', nullable=True)
        assert not (train_aa_embeddings and aa_embedding_path is not None), \
            f"{location}: only one of aa_embedding_path and train_aa_embeddings can be set."

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.vj = vj
        self.require_c_start = require_c_start
        self.aa_embedding_path = aa_embedding_path
        self.train_aa_embeddings = train_aa_embeddings
        self.embed_size = embed_size
        self.word2vec_epochs = word2vec_epochs
        self.word2vec_batch_size = word2vec_batch_size
        self.word2vec_learning_rate = word2vec_learning_rate
        self.word2vec_window_size = word2vec_window_size
        self.device = device

        self._alphabet = set(EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID))
        self._v_genes = []
        self._j_genes = []
        self._embedding_file = None
        self._tcrpeg = None
        self.loss_summary_path = None

    @property
    def _sequence_field(self):
        return get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)

    @property
    def _upstream_max_length(self):
        # the package pads the start token + sequence to max_length, so sequences have to be shorter than max_length;
        # one more position is added so that sampled sequences that did not reach the end token (which the package
        # truncates to max_length - 1 residues) are longer than self.max_length and can be discarded
        return self.max_length + 2

    @staticmethod
    def _set_seed(seed: int):
        import torch
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _make_tcrpeg(self, train_data: list = None, state_dict_file: Path = None):
        from tcrpeg.TCRpeg import TCRpeg as UpstreamTCRpeg

        embedding_size = pd.read_csv(self._embedding_file, header=None, index_col=0).shape[1]
        model = UpstreamTCRpeg(max_length=self._upstream_max_length, embedding_size=embedding_size,
                               hidden_size=self.hidden_size, dropout=self.dropout, num_layers=self.num_layers,
                               device=self.device, load_data=train_data is not None, path_train=train_data,
                               embedding_path=str(self._embedding_file),
                               vs_list=list(self._v_genes) if self.vj else None,
                               js_list=list(self._j_genes) if self.vj else None, vj=self.vj)
        model.create_model(load=state_dict_file is not None,
                           path=str(state_dict_file) if state_dict_file is not None else None, vj=self.vj)
        model.model.max_length = self._upstream_max_length  # not passed on by the package for the non-vj model
        return model

    def fit(self, data: Dataset, path: Path = None):
        self.set_locus(data)
        self._set_seed(self.seed)

        df = self._filter_training_data(data.data.topandas())
        sequences = df[self._sequence_field].tolist()

        if self.vj:
            self._v_genes = sorted(df['v_call'].unique().tolist())
            self._j_genes = sorted(df['j_call'].unique().tolist())
            train_data = [[seq, v, j] for seq, v, j in zip(sequences, df['v_call'], df['j_call'])]
        else:
            train_data = sequences

        self._embedding_file = self._prepare_embeddings(sequences, path)
        self._tcrpeg = self._make_tcrpeg(train_data=train_data)

        batch_size = min(self.batch_size, len(sequences))
        print_log(f"{TCRpeg.__name__}: training on {len(sequences)} sequences with batch size {batch_size}...", True)
        train_func = self._tcrpeg.train_tcrpeg_vj if self.vj else self._tcrpeg.train_tcrpeg
        train_func(epochs=self.num_epochs, batch_size=batch_size, lr=self.learning_rate)
        self._tcrpeg.model.eval()
        print_log(f"{TCRpeg.__name__}: finished training.", True)

    def _prepare_embeddings(self, sequences: list, path: Path = None) -> Path:
        if self.train_aa_embeddings:
            from tcrpeg.word2vec import word2vec

            import tempfile
            emb_dir = PathBuilder.build(path / f'tcrpeg_embeddings_{self.name}') if path is not None \
                else Path(tempfile.mkdtemp())
            emb_file = emb_dir / TCRpeg.EMBEDDING_FILE_NAME
            print_log(f"{TCRpeg.__name__}: learning amino acid embeddings with word2vec...", True)
            word2vec(path=sequences, epochs=self.word2vec_epochs, batch_size=self.word2vec_batch_size,
                     device=self.device, record_path=str(emb_file), lr=self.word2vec_learning_rate,
                     window_size=self.word2vec_window_size, embedding_dims=self.embed_size)
            return emb_file
        elif self.aa_embedding_path is not None:
            emb_file = Path(self.aa_embedding_path)
        else:
            import tcrpeg
            emb_file = Path(tcrpeg.__file__).parent / 'data/embedding_32.txt'

        assert emb_file.is_file(), f"{TCRpeg.__name__}: amino acid embedding file {emb_file} does not exist."
        tokens = ['s'] + sorted(self._alphabet) + ['e']
        emb = pd.read_csv(emb_file, header=None, index_col=0)
        assert emb.index.tolist() == tokens and not emb.isna().any().any(), \
            f"{TCRpeg.__name__}: {emb_file} should have one row per token in this order: {tokens}, followed by the " \
            f"embedding values."
        return emb_file

    def _filter_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        seq_col = self._sequence_field
        original_count = df.shape[0]
        df = df[df[seq_col].notna()].copy()
        df[seq_col] = df[seq_col].astype(str)

        keep = self._is_valid_sequence(df[seq_col])

        if 'productive' in df.columns:
            keep &= ~df['productive'].astype(str).str.upper().isin(['F', 'FALSE'])

        check_c = self.require_c_start and self.region_type == RegionType.IMGT_JUNCTION
        if check_c:
            keep &= df[seq_col].str.startswith('C')

        if self.vj:
            assert 'v_call' in df.columns and 'j_call' in df.columns, \
                f"{TCRpeg.__name__}: v_call and j_call are required in the training data when vj is True."
            for gene_col in ['v_call', 'j_call']:
                keep &= df[gene_col].notna() & (df[gene_col].astype(str).str.len() > 0)

        df = df[keep]
        print_log(f"{TCRpeg.__name__}: removed {original_count - df.shape[0]} out of {original_count} training "
                  f"sequences (empty, longer than {self.max_length}, non-standard amino acids, non-productive"
                  f"{', not starting with C' if check_c else ''}{', missing V or J gene' if self.vj else ''}).", True)

        assert df.shape[0] > 0, f"{TCRpeg.__name__}: no training sequences left after filtering; check region_type, " \
                                f"max_length and require_c_start."
        return df

    def _is_valid_sequence(self, sequences: pd.Series) -> pd.Series:
        return sequences.apply(lambda seq: 0 < len(seq) <= self.max_length and all(aa in self._alphabet for aa in seq))

    def is_same(self, model) -> bool:
        import torch

        if not isinstance(model, TCRpeg) or self._export_params() != model._export_params() \
                or self._v_genes != model._v_genes or self._j_genes != model._j_genes:
            return False
        if self._tcrpeg is None or model._tcrpeg is None:
            return self._tcrpeg is None and model._tcrpeg is None

        other_state = model._tcrpeg.model.state_dict()
        return all(torch.equal(val.cpu(), other_state[key].cpu())
                   for key, val in self._tcrpeg.model.state_dict().items())

    def generate_sequences(self, count: int, seed: int, path: Path, sequence_type: SequenceType, compute_p_gen: bool,
                           max_failed_batches: int = 100):
        if sequence_type != SequenceType.AMINO_ACID:
            raise NotImplementedError(f"{TCRpeg.__name__}: generating {sequence_type.name.lower()} sequences is not "
                                      f"supported, the model generates only amino acid sequences.")
        assert self._tcrpeg is not None, f"{TCRpeg.__name__}: the model has to be fitted or loaded before generation."

        self._set_seed(seed)
        self._tcrpeg.model.eval()
        batch_size = min(self.batch_size, max(count, 1))
        sequences, v_calls, j_calls, failed_batches = [], [], [], 0

        while len(sequences) < count:
            if self.vj:
                seqs, vs, js = self._tcrpeg.generate_tcrpeg_vj(num_to_gen=batch_size, batch_size=batch_size)
            else:
                seqs = self._tcrpeg.generate_tcrpeg(num_to_gen=batch_size, batch_size=batch_size)
                vs, js = [None] * len(seqs), [None] * len(seqs)

            valid = [(s, v, j) for s, v, j in zip(seqs, vs, js) if 0 < len(s) <= self.max_length]
            sequences.extend(el[0] for el in valid)
            v_calls.extend(el[1] for el in valid)
            j_calls.extend(el[2] for el in valid)

            if len(valid) == 0:
                failed_batches += 1
                if failed_batches >= max_failed_batches:
                    raise RuntimeError(f"{TCRpeg.__name__}: could not generate valid sequences (non-empty, with at "
                                       f"most {self.max_length} amino acids) in {max_failed_batches} batches.")

        df = pd.DataFrame({self._sequence_field: sequences[:count]})
        if self.vj:
            df['v_call'], df['j_call'] = v_calls[:count], j_calls[:count]
        if compute_p_gen:
            df['p_gen'] = self._compute_p_gens(df[self._sequence_field].tolist(),
                                               df['v_call'].tolist() if self.vj else None,
                                               df['j_call'].tolist() if self.vj else None)

        df['locus'] = self.locus.to_string() if self.locus is not None else ''
        df['gen_model_name'] = self.name

        print_log(f"{TCRpeg.__name__} {self.name}: generated {count} sequences.", True)

        type_dict = {'gen_model_name': str, **({'p_gen': float} if compute_p_gen else {})}
        return SequenceDataset.build_from_partial_df(df, PathBuilder.build(path), 'synthetic_tcrpeg_dataset',
                                                     {'gen_model_name': [self.name]}, type_dict)

    def compute_p_gens(self, sequences, sequence_type: SequenceType, sequence_field: str = None) -> np.ndarray:
        """
        Computes p(x) for each sequence (including the probability of the end token), or p(x, V, J) if the model was
        trained with vj=True and V and J genes are available in the input. Sequences with non-standard amino acids,
        longer than max_length, or with V or J genes not seen in training get p_gen 0. Sequences can be given as a
        SequenceDataset, a pandas DataFrame, a bionumpy dataclass (e.g., AIRRSequenceSet) or a list of strings.
        """
        if sequence_type != SequenceType.AMINO_ACID:
            raise NotImplementedError(f"{TCRpeg.__name__}: p_gen can only be computed for amino acid sequences.")

        seqs, v_calls, j_calls = self._extract_sequences(sequences, sequence_field)
        return self._compute_p_gens(seqs, v_calls, j_calls)

    def compute_p_gen(self, sequence: dict, sequence_type: SequenceType, sequence_field: str = None) -> float:
        if isinstance(sequence, str):
            sequence = {self._sequence_field: sequence}
        field = sequence_field if sequence_field is not None else \
            (self._sequence_field if self._sequence_field in sequence else 'sequence_aa')
        df = pd.DataFrame({key: [sequence[key]] for key in [field, 'v_call', 'j_call'] if key in sequence})
        return float(self.compute_p_gens(df, sequence_type, field)[0])

    def _extract_sequences(self, sequences, sequence_field: str = None):
        if isinstance(sequences, SequenceDataset):
            sequences = sequences.data
        if isinstance(sequences, (list, tuple, np.ndarray)):
            return [str(s) for s in sequences], None, None

        if isinstance(sequences, pd.DataFrame):
            columns = sequences.columns
            get_col = lambda col: sequences[col].astype(str).tolist()
        else:
            columns = [f for f in [sequence_field, self._sequence_field, 'sequence_aa', 'v_call', 'j_call']
                       if f is not None and hasattr(sequences, f)]
            get_col = lambda col: [str(s) for s in getattr(sequences, col).tolist()]

        field = sequence_field if sequence_field is not None else \
            (self._sequence_field if self._sequence_field in columns else 'sequence_aa')
        assert field in columns, f"{TCRpeg.__name__}: sequence field {field} not found in the input."

        has_genes = self.vj and 'v_call' in columns and 'j_call' in columns
        return get_col(field), get_col('v_call') if has_genes else None, get_col('j_call') if has_genes else None

    def _compute_p_gens(self, sequences: list, v_calls: list = None, j_calls: list = None) -> np.ndarray:
        assert self._tcrpeg is not None, f"{TCRpeg.__name__}: the model has to be fitted or loaded first."

        log_probs = np.full(len(sequences), -np.inf)
        valid = self._is_valid_sequence(pd.Series(sequences, dtype=object)).values.astype(bool)
        if v_calls is not None:
            valid &= np.array([v in self._v_genes and j in self._j_genes for v, j in zip(v_calls, j_calls)],
                              dtype=bool)

        valid_ind = np.where(valid)[0]
        self._tcrpeg.model.eval()

        for start in range(0, len(valid_ind), self.batch_size):
            batch_ind = valid_ind[start: start + self.batch_size]
            seqs = [sequences[i] for i in batch_ind]
            if not self.vj:
                log_probs[batch_ind] = self._tcrpeg.sampling_tcrpeg(seqs)
            else:
                genes = ([v_calls[i] for i in batch_ind], [j_calls[i] for i in batch_ind]) if v_calls is not None \
                    else (None, None)
                log_probs[batch_ind] = self._log_probs_vj(seqs, *genes)

        return np.exp(log_probs)

    def _log_probs_vj(self, seqs: list, v_calls: list = None, j_calls: list = None) -> np.ndarray:
        """
        log p(x) (+ log p(V|x) + log p(J|x) if genes are given) for the vj model; this follows sampling_tcrpeg_vj from
        the TCRpeg GitHub repository, which is not part of the PyPI release (1.0.6), where sampling_tcrpeg does not
        support the vj model
        """
        import torch

        if v_calls is not None and hasattr(self._tcrpeg, 'sampling_tcrpeg_vj'):
            return self._tcrpeg.sampling_tcrpeg_vj([seqs, v_calls, j_calls])

        with torch.no_grad():
            inputs, targets, lengths = self._tcrpeg.aas2embs(seqs)
            inputs, targets, lengths = (torch.LongTensor(inputs).to(self.device),
                                        torch.LongTensor(targets).to(self.device), torch.LongTensor(lengths))
            logp, v_pre, j_pre = self._tcrpeg.model(inputs, lengths)
            target = (targets - 1)[:, :int(lengths.max())]
            mask = target >= 0
            token_log_probs = torch.gather(logp, 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            log_probs = (token_log_probs * mask).sum(dim=1)

            if v_calls is not None:
                v_ind = torch.as_tensor([self._tcrpeg.v2idx[v] for v in v_calls], device=self.device)
                j_ind = torch.as_tensor([self._tcrpeg.j2idx[j] for j in j_calls], device=self.device)
                log_probs += torch.log_softmax(v_pre, -1).gather(1, v_ind.unsqueeze(1)).squeeze(1)
                log_probs += torch.log_softmax(j_pre, -1).gather(1, j_ind.unsqueeze(1)).squeeze(1)

        return log_probs.cpu().numpy()

    def can_compute_p_gens(self) -> bool:
        return True

    def can_generate_from_skewed_gene_models(self) -> bool:
        return False

    def generate_from_skewed_gene_models(self, v_genes: list, j_genes: list, seed: int, path: Path,
                                         sequence_type: SequenceType, batch_size: int, compute_p_gen: bool):
        raise NotImplementedError(f"{TCRpeg.__name__}: generating sequences from skewed gene models is not supported.")

    def _export_params(self) -> dict:
        return {'locus': self.locus.name if self.locus is not None else None,
                'region_type': self.region_type.name, 'hidden_size': self.hidden_size, 'num_layers': self.num_layers,
                'dropout': self.dropout, 'num_epochs': self.num_epochs, 'batch_size': self.batch_size,
                'learning_rate': self.learning_rate, 'max_length': self.max_length, 'vj': self.vj,
                'require_c_start': self.require_c_start, 'aa_embedding_path': self.aa_embedding_path,
                'train_aa_embeddings': self.train_aa_embeddings, 'embed_size': self.embed_size,
                'word2vec_epochs': self.word2vec_epochs, 'word2vec_batch_size': self.word2vec_batch_size,
                'word2vec_learning_rate': self.word2vec_learning_rate,
                'word2vec_window_size': self.word2vec_window_size, 'device': self.device, 'name': self.name,
                'seed': self.seed}

    def save_model(self, path: Path) -> Path:
        import torch

        model_path = PathBuilder.build(path / 'model')

        write_yaml(filename=model_path / 'model_overview.yaml',
                   yaml_dict={'type': self.__class__.__name__, **self._export_params(),
                              'genes': {'v': list(self._v_genes), 'j': list(self._j_genes)}})

        torch.save({k: v.cpu() for k, v in self._tcrpeg.model.state_dict().items()},
                   model_path / TCRpeg.STATE_DICT_FILE_NAME)
        if Path(self._embedding_file).resolve() != (model_path / TCRpeg.EMBEDDING_FILE_NAME).resolve():
            shutil.copy(self._embedding_file, model_path / TCRpeg.EMBEDDING_FILE_NAME)

        return Path(shutil.make_archive(str(path / f'trained_model_{self.name}'), 'zip', str(model_path))).absolute()
