import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from glmnet import LogitNet

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.bnp_util import write_yaml
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.util.Util import Util
from immuneML.util.PathBuilder import PathBuilder

MAX_TORCH_ITERS = 200


class _TorchLogReg(BaseEstimator, ClassifierMixin):
    """sklearn-compatible logistic regression via PyTorch LBFGS with a per-feature penalty mask."""

    def __init__(self, C=1.0, alpha=1.0, non_penalized_indices=None, max_iter=200, device='cpu'):
        self.C = C
        self.alpha = alpha
        self.non_penalized_indices = non_penalized_indices
        self.max_iter = max_iter
        self.device = device

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        try:
            self._fit_on_device(X_arr, y, self.device)
        except RuntimeError as e:
            if 'cuda' in str(self.device) and Util.is_gpu_oom_error(e):
                self._fall_back_to_cpu(e, during="fitting")
                self._fit_on_device(X_arr, y, self.device)
            else:
                raise
        self.classes_ = np.unique(y)
        return self

    def _fit_on_device(self, X_arr, y, device):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        n, p = X_arr.shape
        dev = torch.device(device)
        X_t = torch.from_numpy(X_arr).float().to(dev)
        y_t = torch.from_numpy(y.astype(float)).float().unsqueeze(1).to(dev)

        mask = torch.ones(p, device=dev)
        if self.non_penalized_indices:
            mask[list(self.non_penalized_indices)] = 0.0

        self.linear_ = nn.Linear(p, 1).to(dev)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS(self.linear_.parameters(), lr=1.0,
                                 max_iter=self.max_iter, line_search_fn='strong_wolfe')
        reg_scale = 1.0 / max(self.C, 1e-8)
        alpha = float(self.alpha)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.linear_(X_t), y_t)
            w = self.linear_.weight.squeeze()
            penalty = alpha * torch.sum(mask * torch.abs(w)) + \
                      (1.0 - alpha) * 0.5 * torch.sum(mask * w ** 2)
            total = loss + reg_scale * penalty
            total.backward()
            return total

        optimizer.step(closure)

    def predict_proba(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        try:
            probs = self._predict_proba_on_device(X_arr, self.device)
        except RuntimeError as e:
            if 'cuda' in str(self.device) and Util.is_gpu_oom_error(e):
                self._fall_back_to_cpu(e, during="inference")
                self.linear_ = self.linear_.to('cpu')
                probs = self._predict_proba_on_device(X_arr, self.device)
            else:
                raise
        return np.column_stack([1.0 - probs, probs])

    def _predict_proba_on_device(self, X_arr, device):
        import torch
        dev = torch.device(device)
        X_t = torch.from_numpy(X_arr.astype(np.float32)).to(dev)
        with torch.no_grad():
            probs = torch.sigmoid(self.linear_(X_t)).cpu().numpy().ravel()
        return probs

    def _fall_back_to_cpu(self, error: Exception, during: str):
        import torch
        logging.warning(f"_TorchLogReg: GPU ran out of memory during {during} ({error}). "
                         f"Falling back to CPU (device='cpu') and retrying.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = 'cpu'

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def _compute_lambda_sequence(X, y, alpha, n_lambda, min_lambda_ratio=None):
    """Log-spaced lambda sequence matching glmnet's automatic path for logistic regression.

    At beta=0 the gradient of the log-loss is X^T(y - 0.5) / n, so lambda_max is the
    smallest lambda that shrinks all penalised coefficients to zero.
    """
    X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    n, p = X_arr.shape
    grad = np.abs(X_arr.T @ (y - y.mean())) / n
    lambda_max = grad.max() / max(float(alpha), 1e-3)
    if min_lambda_ratio is None:
        min_lambda_ratio = 1e-4 if n >= p else 1e-2
    lambda_min = lambda_max * min_lambda_ratio
    return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambda))


def _lbfgs_step(linear, X_t, y_t, mask, lam, alpha, max_iter):
    """One warm-started LBFGS step along the regularisation path for a single lambda value."""
    import torch.nn as nn
    import torch.optim as optim
    import torch

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(linear.parameters(), lr=1.0,
                              max_iter=max_iter, line_search_fn='strong_wolfe')
    _lam, _alpha = float(lam), float(alpha)

    def closure():
        optimizer.zero_grad()
        loss = criterion(linear(X_t), y_t)
        w = linear.weight.squeeze()
        penalty = _alpha * torch.sum(mask * torch.abs(w)) + \
                  (1.0 - _alpha) * 0.5 * torch.sum(mask * w ** 2)
        total = loss + _lam * penalty
        total.backward()
        return total

    optimizer.step(closure)


class LogRegressionCustomPenalty(MLMethod):
    """
    Logistic Regression with custom penalty factors for specific features.

    **Specification arguments**:

    - non_penalized_features (list): List of feature names that should not be penalized.

    - non_penalized_encodings (list): List of encoding names (class names) whose features should not be penalized. This
      parameter can be used only in combination with CompositeEncoder. None of the features from the specified encodings
      will be penalized. If both non_penalized_features and non_penalized_encodings are provided, the union of the two
      will be used.

    - backend (str): 'glmnet' (default) or 'torch'. Both backends use the same regularisation parameters below.
      The torch backend uses PyTorch LBFGS with a warm-started regularisation path and stratified K-fold CV
      to select lambda, replicating glmnet's path-fitting strategy.

    - alpha (float): Elastic net mixing parameter. 0 = ridge, 1 = lasso. Default 1.

    - n_lambda (int): Number of lambda values in the regularisation path. Default 100.

    - min_lambda_ratio (float): Ratio of the smallest to largest lambda. Default None (auto: 1e-4 if n>=p, else 1e-2).
      Note: glmnet names this parameter min_lambda_ratio as well.

    - n_splits (int): Cross-validation folds for lambda selection. Default 3.

    - scoring (str): Scoring metric for lambda selection when using the glmnet backend; accepts any scikit-learn
      scorer name (e.g. 'accuracy', 'roc_auc', 'average_precision'). Default None (accuracy). Under strong class
      imbalance, 'average_precision' (PR-AUC) can be a more sensitive criterion for choosing lambda than 'roc_auc',
      since ROC-AUC is prevalence-invariant while PR-AUC's baseline tracks the positive-class prevalence directly.
      Ignored by the torch backend.

    - pos_weight: up-weights the positive class when fitting with the glmnet backend, passed through as glmnet's
      `sample_weight` (rows with the positive class get this weight, all other rows get weight 1). Not supported by
      the torch backend (raises an error if set together with backend='torch'). One of: None (default, no
      reweighting, i.e. the previous behaviour), 'balanced' (weight is set to (number of negative examples) /
      (number of positive examples), computed from the training data being fit), or a positive number used directly
      as the weight.

    - random_state (int): Random seed for CV fold splitting.

    - max_iter (int): Maximum solver iterations. For glmnet this is coordinate descent passes (default 100000);
      for torch this is LBFGS iterations per warm-started path step (capped at 50 internally — warm starting
      means each step already starts near the solution and needs very few iterations).

    Additional keyword arguments are forwarded to LogitNet when using the glmnet backend (e.g. standardize,
    fit_intercept, cut_point, lambda_path). They are ignored by the torch backend.

    **YAML specification:**

    .. code-block:: yaml

        ml_methods:
            custom_log_reg:
                LogRegressionCustomPenalty:
                    backend: torch          # 'glmnet' (default) or 'torch'
                    alpha: 1                # 1 for lasso, 0 for ridge
                    n_lambda: 100
                    n_splits: 3
                    non_penalized_features: []
                    non_penalized_encodings: ['Metadata']
                    random_state: 42

    """

    def __init__(self, non_penalized_features: list = None, name: str = None, label: Label = None,
                 non_penalized_encodings: list = None, backend: str = None,
                 alpha: float = None, n_lambda: int = None, min_lambda_ratio: float = None,
                 n_splits: int = None, scoring: str = None, pos_weight=None, random_state: int = None,
                 max_iter: int = None, device: str = None, **kwargs):
        super().__init__(name=name, label=label)
        self.non_penalized_features = non_penalized_features if non_penalized_features is not None else []
        self.non_penalized_encodings = non_penalized_encodings if non_penalized_encodings is not None else []
        self.backend = backend
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.n_splits = n_splits
        self.scoring = scoring
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.device = device
        self.kwargs = kwargs  # forwarded to LogitNet for glmnet backend

        for ind, encoding in enumerate(self.non_penalized_encodings):
            if 'Encoder' not in encoding:
                self.non_penalized_encodings[ind] = encoding + 'Encoder'

        if backend == 'torch':
            if pos_weight is not None:
                raise ValueError(f"{self.__class__.__name__}: pos_weight is currently only supported with "
                                 f"backend='glmnet', not with the torch backend.")
            try:
                import torch
            except ImportError:
                raise ImportError("LogRegressionCustomPenalty: PyTorch is required for the 'torch' backend. "
                                  "Please install it to use this option.")

        self.model = None
        self.feature_names = None

    def _resolve_non_penalized_features(self, encoded_data: EncodedData):
        if encoded_data.encoding == 'CompositeEncoder' and self.non_penalized_encodings:
            from_encodings = encoded_data.feature_annotations[
                encoded_data.feature_annotations['encoder'].isin(self.non_penalized_encodings)
            ]['feature'].tolist()
            self.non_penalized_features = list(set(from_encodings) | set(self.non_penalized_features))
            logging.info(f"{self.__class__.__name__}: inferred non-penalized features: {self.non_penalized_features}")

    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        X = encoded_data.examples
        y = Util.map_to_new_class_values(encoded_data.labels[self.label.name], self.class_mapping)
        self.feature_names = encoded_data.feature_names

        self._resolve_non_penalized_features(encoded_data)

        non_penalized_indices = [i for i, f in enumerate(self.feature_names) if f in self.non_penalized_features]
        penalty_factor = np.ones(X.shape[1])
        for idx in non_penalized_indices:
            penalty_factor[idx] = 0.0

        if self.backend == 'glmnet':
            glmnet_params = dict(
                alpha=self.alpha, n_lambda=self.n_lambda, n_splits=self.n_splits,
                random_state=self.random_state, max_iter=self.max_iter,
                n_jobs=cores_for_training, **self.kwargs
            )
            if self.min_lambda_ratio is not None:
                glmnet_params['min_lambda_ratio'] = self.min_lambda_ratio
            if self.scoring is not None:
                glmnet_params['scoring'] = self.scoring
            self.model = LogitNet(**glmnet_params)
            sample_weight = self._resolve_sample_weight(y)
            self.model.fit(X, y, sample_weight=sample_weight, relative_penalties=penalty_factor)
        elif self.backend == 'torch':
            try:
                self._fit_torch(X, y, non_penalized_indices, cores_for_training)
            except RuntimeError as e:
                if 'cuda' in str(self.device) and Util.is_gpu_oom_error(e):
                    import torch
                    logging.warning(f"{self.__class__.__name__}: GPU ran out of memory while fitting ({e}). "
                                     f"Falling back to CPU (device='cpu') and retrying.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.device = 'cpu'
                    self._fit_torch(X, y, non_penalized_indices, cores_for_training)
                else:
                    raise
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'glmnet' or 'torch'.")

    def _resolve_sample_weight(self, y):
        """Resolves self.pos_weight into a per-example sample_weight array for glmnet's LogitNet.fit(), computed
        from the labels y of the training data currently being fit. Rows with the positive class (y == 1) get the
        resolved weight, all other rows get weight 1. Returns None (i.e. uniform weighting, glmnet's default) when
        self.pos_weight is None."""
        if self.pos_weight is None:
            return None

        if isinstance(self.pos_weight, str):
            assert self.pos_weight == 'balanced', \
                f"{self.__class__.__name__}: pos_weight has to be None, 'balanced', or a positive number, " \
                f"got '{self.pos_weight}' instead."
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            if n_pos == 0 or n_neg == 0:
                logging.warning(f"{self.__class__.__name__}: cannot compute a 'balanced' pos_weight since one of "
                                f"the classes is not present in the data being fit; fitting without pos_weight instead.")
                return None
            weight_value = n_neg / n_pos
        else:
            weight_value = float(self.pos_weight)

        return np.where(y == 1, weight_value, 1.0)

    def _fit_torch(self, X, y, non_penalized_indices, n_jobs):
        # With warm starting each step starts near the solution, so few LBFGS iterations suffice.
        # Cap at MAX_TORCH_ITERS; the glmnet default of 100000 is for coordinate descent, not quasi-Newton.
        import torch
        from torch import nn
        lbfgs_max_iter = min(self.max_iter, MAX_TORCH_ITERS)
        alpha = float(self.alpha)
        dev = torch.device(self.device)

        X_dense = (X.toarray() if hasattr(X, 'toarray') else np.asarray(X)).astype(np.float32)
        lambdas = _compute_lambda_sequence(X_dense, y, alpha, self.n_lambda, self.min_lambda_ratio)
        n, p = X_dense.shape

        mask = torch.ones(p, device=dev)
        if non_penalized_indices:
            mask[list(non_penalized_indices)] = 0.0

        # K-fold CV with a warm-started path per fold to score each lambda
        cv_scores = np.zeros(self.n_lambda)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X_dense, y):
            X_tr = torch.from_numpy(X_dense[train_idx]).to(dev)
            y_tr = torch.from_numpy(y[train_idx].astype(np.float32)).unsqueeze(1).to(dev)
            X_val = torch.from_numpy(X_dense[val_idx]).to(dev)
            y_val = y[val_idx]

            linear = nn.Linear(p, 1).to(dev)
            for i, lam in enumerate(lambdas):
                _lbfgs_step(linear, X_tr, y_tr, mask, lam, alpha, lbfgs_max_iter)
                with torch.no_grad():
                    probs = torch.sigmoid(linear(X_val)).cpu().numpy().ravel()
                try:
                    cv_scores[i] += roc_auc_score(y_val, probs)
                except ValueError:
                    cv_scores[i] += 0.5

        best_idx = int(np.argmax(cv_scores / self.n_splits))

        # Refit on full data along the path up to the selected lambda
        X_t = torch.from_numpy(X_dense).to(dev)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(dev)
        linear_final = nn.Linear(p, 1).to(dev)
        for lam in lambdas[:best_idx + 1]:
            _lbfgs_step(linear_final, X_t, y_t, mask, lam, alpha, lbfgs_max_iter)

        self.model = _TorchLogReg(C=float(1.0 / lambdas[best_idx]), alpha=alpha,
                                   non_penalized_indices=non_penalized_indices,
                                   max_iter=lbfgs_max_iter, device=self.device)
        self.model.linear_ = linear_final
        self.model.classes_ = np.unique(y)

    def _predict(self, encoded_data: EncodedData):
        predictions = self.model.predict(encoded_data.examples)
        return {self.label.name: Util.map_to_old_class_values(np.array(predictions), self.class_mapping)}

    def _predict_proba(self, encoded_data: EncodedData):
        class_names = Util.map_to_old_class_values(self.model.classes_, self.class_mapping)
        probabilities = self.model.predict_proba(encoded_data.examples)
        return {self.label.name: {class_name: probabilities[:, i] for i, class_name in enumerate(class_names)}}

    def store(self, path: Path):
        PathBuilder.build(path)
        write_yaml(path / 'model.yaml', vars(self))
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'non_penalized_features': self.non_penalized_features,
                'feature_names': self.feature_names
            }, f)

    def load(self, path: Path):
        with open(path / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
            self.model = model['model']
            self.non_penalized_features = model['non_penalized_features']
            self.feature_names = model['feature_names']

    def get_params(self, for_refitting=False) -> dict:
        params = {
            'alpha': self.alpha,
            'non_penalized_features': self.non_penalized_features,
            'backend': self.backend,
            'pos_weight': self.pos_weight,
        }
        if self.backend == 'glmnet':
            params.update({
                'lambda_best': self.model.lambda_best_,
                'lambda_max': self.model.lambda_max_,
                'random_state': self.model.random_state,
            })
            if not for_refitting:
                params['coefficients'] = self.model.coef_[0].tolist()
        else:
            params['C'] = self.model.C
            if not for_refitting:
                params['coefficients'] = self.model.linear_.weight.squeeze().detach().cpu().numpy().tolist()
        return params

    def can_predict_proba(self) -> bool:
        return True

    def can_fit_with_example_weights(self) -> bool:
        return False

    def get_compatible_encoders(self):
        from immuneML.encodings.diversity_encoding.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
        from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
        from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
        from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
        from immuneML.encodings.protein_embedding.ESMCEncoder import ESMCEncoder
        from immuneML.encodings.protein_embedding.ProtT5Encoder import ProtT5Encoder
        from immuneML.encodings.protein_embedding.TCRBertEncoder import TCRBertEncoder
        from immuneML.encodings.diversity_encoding.ShannonDiversityEncoder import ShannonDiversityEncoder
        from immuneML.encodings.composite_encoding.CompositeEncoder import CompositeEncoder
        from immuneML.encodings.baseline_encoding.GeneFrequencyEncoder import GeneFrequencyEncoder
        from immuneML.encodings.baseline_encoding.MetadataEncoder import MetadataEncoder
        from immuneML.encodings.amino_acid_property_encoding.AminoAcidPropertyEncoder import AminoAcidPropertyEncoder
        from immuneML.encodings.sequence_length_encoding.SequenceLengthEncoder import SequenceLengthEncoder

        from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import \
            CompAIRRSequenceAbundanceEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, EvennessProfileEncoder,
                MatchedSequencesEncoder, MatchedReceptorsEncoder, MatchedRegexEncoder, MotifEncoder,
                ESMCEncoder, ProtT5Encoder, TCRBertEncoder, ShannonDiversityEncoder, CompAIRRSequenceAbundanceEncoder,
                CompositeEncoder, GeneFrequencyEncoder, MetadataEncoder, AminoAcidPropertyEncoder,
                SequenceLengthEncoder]
