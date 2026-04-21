import logging
from pathlib import Path
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet, AminoAcidXEncoding
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset, ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator

_FACTORS_DIR = Path(__file__).parent.parent.parent / "config" / "physicochemical_factors"
SUPPORTED_FACTORS = ['atchley', 'amino_acid_property', 'kidera']


class AminoAcidPropertyEncoder(DatasetEncoder):
    """
    Encodes a dataset by replacing each amino acid in a sequence with its biophysicochemical
    factor vector and averaging those vectors across all positions in the sequence.
    Three factor sets are supported, each stored as a TSV file under
    ``immuneML/config/physicochemical_factors/``:

    - ``atchley`` — 5 factors per amino acid (Atchley et al., 2005).
    - ``kidera`` — 10 factors per amino acid (Kidera et al., 1985).
    - ``amino_acid_property`` — 14 mixed physicochemical descriptors per amino acid compiled
      from several published sources and originally aggregated in VDJtools (Shugay et al., 2015).

    Characters outside the standard 20-amino-acid alphabet (gaps, X, etc.) are silently
    skipped; a sequence with no known amino acids is encoded as an all-zero vector.

    For SequenceDatasets the output shape is ``[n_sequences, n_factors]``. For ReceptorDatasets
    each chain is encoded independently and the resulting vectors are concatenated (chains ordered
    alphabetically by locus name), giving shape ``[n_receptors, 2 × n_factors]``.

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets


    **Specification arguments:**

    - factors (str): Which set of biophysicochemical factors to use. Valid values: ``atchley``
      (5 factors), ``kidera`` (10 factors), or ``amino_acid_property`` (14 factors).

    - region_type (str): Which part of the receptor sequence to encode (e.g. ``imgt_cdr3``).

    - scale_to_zero_mean (bool): Whether to scale each feature to zero mean across examples
      after encoding. Defaults to ``true``.

    - scale_to_unit_variance (bool): Whether to scale each feature to unit variance across
      examples after encoding. Defaults to ``true``.


    **References:**

    Factor values downloaded from `vadimnazarov/kidera-atchley
    <https://github.com/vadimnazarov/kidera-atchley>`_.

    - W.R. Atchley, J. Zhao, A.D. Fernandes,  & T. Drüke,   Solving the protein sequence metric problem, Proc.
      Natl. Acad. Sci. U.S.A. 102 (18) 6395-6400, https://doi.org/10.1073/pnas.0408677102 (2005).

    - Kidera, A., Konishi, Y., Oka, M. et al. Statistical analysis of the physical properties of the 20 naturally
      occurring amino acids. J Protein Chem 4, 23–55 (1985). https://doi.org/10.1007/BF01025492

    - Shugay M et al. VDJtools: Unifying Post-analysis of T Cell Receptor Repertoires. PLoS Comp Biol 2015;
      11(11):e1004503-e1004503.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_atchley_encoder:
                    AminoAcidProperty:
                        factors: atchley
                        region_type: imgt_cdr3
                        scale_to_zero_mean: true
                        scale_to_unit_variance: true

                my_kidera_encoder:
                    AminoAcidProperty:
                        factors: kidera
                        region_type: imgt_cdr3

                my_aa_property_encoder:
                    AminoAcidProperty:
                        factors: amino_acid_property
                        region_type: imgt_cdr3
    """

    def __init__(self, factors: str, region_type: RegionType, scale_to_zero_mean: bool = False,
                 scale_to_unit_variance: bool = False, name: str = None):
        super().__init__(name=name)
        self.factors = factors
        self.region_type = region_type
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance
        self.factor_table, self.factor_names = self._load_factor_file(factors)
        self.scaler = None
        self._lookup, self._valid_mask = self._build_lookup_table()

    @staticmethod
    def _load_factor_file(factors: str) -> tuple:
        """Load a TSV factor file and return (factor_table dict, factor_names list)."""
        filepath = _FACTORS_DIR / f"{factors}.tsv"
        factor_table = {}
        with open(filepath) as fh:
            header = next(fh).rstrip('\n').split('\t')
            factor_names = header[1:]
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                aa = parts[0]
                factor_table[aa] = [float(v) for v in parts[1:]]
        return factor_table, factor_names

    @staticmethod
    def build_object(dataset: Dataset, **params):
        location = AminoAcidPropertyEncoder.__name__
        ParameterValidator.assert_in_valid_list(params.get('factors'), SUPPORTED_FACTORS,
                                                location, 'factors')
        ParameterValidator.assert_region_type(params, location)
        ParameterValidator.assert_type_and_value(params['scale_to_zero_mean'], bool, location,
                                                 'scale_to_zero_mean')
        ParameterValidator.assert_type_and_value(params['scale_to_unit_variance'], bool, location,
                                                 'scale_to_unit_variance')
        return AminoAcidPropertyEncoder(
            factors=params['factors'],
            region_type=RegionType[params['region_type'].upper()],
            scale_to_zero_mean=params['scale_to_zero_mean'],
            scale_to_unit_variance=params['scale_to_unit_variance'],
            name=params.get('name'),
        )

    def encode(self, dataset: Dataset, params: EncoderParams) -> Dataset:
        cache_params = self._get_caching_params(dataset, params)
        if isinstance(dataset, SequenceDataset):
            return CacheHandler.memo_by_params(cache_params,
                                               lambda: self._encode_sequence_dataset(dataset, params))
        elif isinstance(dataset, ReceptorDataset):
            return CacheHandler.memo_by_params(cache_params,
                                               lambda: self._encode_receptor_dataset(dataset, params))
        elif isinstance(dataset, RepertoireDataset):
            return CacheHandler.memo_by_params(cache_params,
                                               lambda: self._encode_repertoire_dataset(dataset, params))
        else:
            raise RuntimeError(f"{self.__class__.__name__}: {self.name}: unsupported dataset type "
                               f"'{type(dataset).__name__}'.")

    # ------------------------------------------------------------------
    # Core encoding: one averaged factor vector per sequence
    # ------------------------------------------------------------------

    def _build_lookup_table(self) -> tuple:
        """Build factor lookup and validity mask indexed by AminoAcidXEncoding integer codes.

        AminoAcidXEncoding uses alphabet 'ACDEFGHIKLMNPQRSTVWXY*' (codes 0–21).
        The lookup table has one row per code; unknown/non-standard characters (X, *)
        get an all-zero row and are excluded from the per-sequence average.
        """
        alphabet = bytes(AminoAcidXEncoding._alphabet).decode('ascii')
        n_factors = len(next(iter(self.factor_table.values())))
        lookup = np.zeros((len(alphabet), n_factors), dtype=float)
        valid = np.zeros(len(alphabet), dtype=bool)
        for i, aa in enumerate(alphabet):
            if aa in self.factor_table:
                lookup[i] = self.factor_table[aa]
                valid[i] = True
        return lookup, valid

    def _encode_sequence_set(self, sequence_set: AIRRSequenceSet, seq_field: str) -> np.ndarray:
        """Return a ``[n_sequences, n_factors]`` array of per-sequence factor averages.

        Uses the bionumpy EncodedRaggedArray directly to avoid Python-level string
        operations: integer codes are retrieved via ``.ravel().raw()`` and lengths via
        ``.lengths``, then a single ``np.add.reduceat`` aggregates all sequences at once.
        """
        seq_array = getattr(sequence_set, seq_field)  # EncodedRaggedArray
        n_seqs = len(seq_array)
        n_factors = self._lookup.shape[1]

        if n_seqs == 0:
            return np.zeros((0, n_factors), dtype=float)

        lengths = np.asarray(seq_array.lengths)    # [n_seqs]
        flat_codes = seq_array.ravel().raw()        # [total_chars] int codes 0–21

        flat_factors = self._lookup[flat_codes]     # [total_chars, n_factors]
        flat_valid = self._valid_mask[flat_codes]   # [total_chars] bool

        starts = np.concatenate([[0], np.cumsum(lengths[:-1])])

        factor_sums = np.add.reduceat(flat_factors, starts)               # [n_seqs, n_factors]
        valid_counts = np.add.reduceat(flat_valid.astype(float), starts)  # [n_seqs]

        # np.add.reduceat returns a[i] rather than 0 for zero-length slices; fix those.
        zero_len = lengths == 0
        if zero_len.any():
            factor_sums[zero_len] = 0.0
            valid_counts[zero_len] = 0.0
            for i in np.where(zero_len)[0]:
                logging.warning(f"{self.__class__.__name__}: sequence at index {i} is empty; "
                                 f"encoding as zeros.")

        no_valid_aa = (valid_counts == 0) & ~zero_len
        if no_valid_aa.any():
            for i in np.where(no_valid_aa)[0]:
                logging.warning(f"{self.__class__.__name__}: sequence at index {i} contains no "
                                 f"known amino acids; encoding as zeros.")

        safe_counts = np.where(valid_counts == 0, 1.0, valid_counts)
        return factor_sums / safe_counts[:, np.newaxis]

    # ------------------------------------------------------------------
    # Dataset-type-specific encoding methods
    # ------------------------------------------------------------------

    def _encode_sequence_dataset(self, dataset: SequenceDataset, params: EncoderParams) -> SequenceDataset:
        seq_field = get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)
        examples = self._encode_sequence_set(dataset.data, seq_field)
        examples = self._scale_examples(examples, params)

        labels = ({label.name: getattr(dataset.data, label.name).tolist()
                   for label in params.label_config.get_label_objects()}
                  if params.encode_labels else None)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=examples,
            labels=labels,
            example_ids=dataset.data.sequence_id.tolist(),
            feature_names=self._get_feature_names(),
            encoding=AminoAcidPropertyEncoder.__name__,
        )
        return encoded_dataset

    def _encode_receptor_dataset(self, dataset: ReceptorDataset, params: EncoderParams) -> ReceptorDataset:
        seq_field = get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)
        data = dataset.data

        receptor_ids, loci, mask1, mask2 = EncoderHelper.get_receptor_chain_masks(dataset)

        per_seq_embeddings = self._encode_sequence_set(data, seq_field)  # [n_chains, n_factors]

        examples = np.hstack([per_seq_embeddings[mask1], per_seq_embeddings[mask2]])  # [n_receptors, 2*n_factors]
        examples = self._scale_examples(examples, params)

        if params.encode_labels:
            df = data.topandas()
            labels = {name: df[name].values[mask1].tolist()
                      for name in params.label_config.get_labels_by_name()}
        else:
            labels = None

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=examples,
            labels=labels,
            example_ids=receptor_ids,
            feature_names=self._get_receptor_feature_names(loci),
            encoding=AminoAcidPropertyEncoder.__name__,
        )
        return encoded_dataset

    def _encode_repertoire_dataset(self, dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        seq_field = get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)

        examples = []
        for repertoire in dataset.repertoires:
            rep_vec = CacheHandler.memo_by_params(
                (repertoire.identifier, self.__class__.__name__, self.factors, self.region_type.name),
                lambda: self._encode_sequence_set(repertoire.data, seq_field).mean(axis=0),
            )
            examples.append(rep_vec)

        examples = self._scale_examples(np.array(examples), params)

        labels = (dataset.get_metadata(params.label_config.get_labels_by_name())
                  if params.encode_labels else None)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=examples,
            labels=labels,
            example_ids=dataset.get_example_ids(),
            feature_names=self._get_feature_names(),
            encoding=AminoAcidPropertyEncoder.__name__,
        )
        return encoded_dataset

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def _scale_examples(self, examples: np.ndarray, params: EncoderParams) -> np.ndarray:
        if not self.scale_to_zero_mean and not self.scale_to_unit_variance:
            return examples
        if params.learn_model:
            self.scaler = StandardScaler(with_mean=self.scale_to_zero_mean,
                                         with_std=self.scale_to_unit_variance)
            return FeatureScaler.standard_scale_fit(self.scaler, examples,
                                                    with_mean=self.scale_to_zero_mean)
        else:
            return FeatureScaler.standard_scale(self.scaler, examples,
                                                with_mean=self.scale_to_zero_mean)

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def _get_feature_names(self) -> List[str]:
        return [f'{self.factors}_{name}' for name in self.factor_names]

    def _get_receptor_feature_names(self, loci: List[str]) -> List[str]:
        return [f'{locus}_{self.factors}_{name}'
                for locus in loci
                for name in self.factor_names]

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _get_caching_params(self, dataset: Dataset, params: EncoderParams, step: str = None) -> tuple:
        return (dataset.identifier,
                tuple(params.label_config.get_labels_by_name()),
                self.factors,
                self.region_type.name,
                self.scale_to_zero_mean,
                self.scale_to_unit_variance,
                params.learn_model,
                step)