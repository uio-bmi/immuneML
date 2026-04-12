import logging
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
from immuneML.util.ParameterValidator import ParameterValidator


class AminoAcidPropertyEncoder(DatasetEncoder):
    """
    Encodes a dataset by replacing each amino acid in a sequence with its biophysicochemical
    factor vector and averaging those vectors across all positions in the sequence.
    Two classical factor sets are supported: Atchley factors (5 factors per amino acid) and
    Kidera factors (10 factors per amino acid). Characters outside the standard 20-amino-acid
    alphabet (gaps, X, etc.) are silently skipped; a sequence with no known amino acids is
    encoded as an all-zero vector.

    For SequenceDatasets the output shape is ``[n_sequences, n_factors]``. For ReceptorDatasets
    each chain is encoded independently and the resulting vectors are concatenated (chains ordered
    alphabetically by locus name), giving shape ``[n_receptors, 2 × n_factors]``. For
    RepertoireDatasets each repertoire is represented by the mean of its per-sequence averaged
    factor vectors, giving shape ``[n_repertoires, n_factors]``.

    **Dataset type:**

    - SequenceDatasets

    - ReceptorDatasets

    - RepertoireDatasets


    **Specification arguments:**

    - factors (str): Which set of biophysicochemical factors to use. Valid values: ``atchley``
      (5 factors) or ``kidera`` (10 factors).

    - region_type (str): Which part of the receptor sequence to encode (e.g. ``imgt_cdr3``).

    - scale_to_zero_mean (bool): Whether to scale each feature to zero mean across examples
      after encoding. Defaults to ``False``.

    - scale_to_unit_variance (bool): Whether to scale each feature to unit variance across
      examples after encoding. Defaults to ``False``.


    **References:**

    - Atchley et al. (2005). Solving the protein sequence metric problem. *PNAS*, 102(18),
      6395–6400.

    - Kidera et al. (1985). Statistical analysis of the physical properties of the 20 naturally
      occurring amino acids. *Journal of Protein Chemistry*, 4(1), 23–55.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_atchley_encoder:
                    AminoAcidProperty:
                        factors: atchley
                        region_type: imgt_cdr3
                        scale_to_zero_mean: False
                        scale_to_unit_variance: False

                my_kidera_encoder:
                    AminoAcidProperty:
                        factors: kidera
                        region_type: imgt_cdr3
    """

    # Atchley et al. (2005), PNAS 102(18): 6395-6400, Table 1.
    # Factors: polarity, propensity for secondary structures, molecular volume,
    #          codon diversity, electrostatic charge
    ATCHLEY_FACTORS = {
        'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
        'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
        'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
        'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
        'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
        'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
        'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
        'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
        'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
        'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
        'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
        'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
        'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
        'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
        'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
        'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
        'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
        'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
    }

    # Kidera et al. (1985), J. Protein Chem. 4(1): 23-55.
    # Factors: helix/bend preference, side-chain size, extended structure preference,
    #          hydrophobicity, double-bend preference, partial specific volume,
    #          flat extended preference, occurrence in alpha region, pK-C,
    #          surrounding hydrophobicity
    KIDERA_FACTORS = {
        'A': [ 0.24, -2.32,  0.60, -0.14,  1.89,  0.22, -0.60, -1.14, -0.09, -0.35],
        'C': [ 0.84, -1.67,  3.71,  0.18, -2.65,  0.00,  1.20,  1.37, -1.68, -0.67],
        'D': [-0.98, -1.43, -3.71, -0.15, -0.86, -0.01, -0.07, -0.29,  0.28, -0.01],
        'E': [-0.77, -1.53, -2.29, -0.35,  2.82, -0.56,  0.03,  0.32, -0.47, -0.14],
        'F': [ 1.06,  1.77,  0.27, -1.43,  1.13,  0.54, -0.49, -0.06,  0.78,  0.04],
        'G': [ 2.05, -4.06,  0.36, -0.82, -0.38,  1.03,  0.27,  0.20,  0.21,  0.12],
        'H': [-0.26,  0.53, -1.87, -1.22, -0.55, -0.87,  0.20,  1.82,  0.32, -0.07],
        'I': [-0.11, -2.02,  3.36, -0.75,  0.55,  0.07,  1.52,  0.95, -0.46, -0.97],
        'K': [-1.18,  1.40,  0.80, -0.75,  0.00, -0.31, -0.05, -0.28,  0.34,  0.74],
        'L': [ 1.22, -2.32,  3.99, -0.27,  0.77,  1.16,  0.83, -0.98,  0.65,  0.09],
        'M': [ 1.12, -2.99,  3.03,  0.00, -0.43,  0.96,  0.82, -0.69, -0.30, -0.32],
        'N': [-0.94, -0.34, -2.29, -1.52, -0.49, -0.85, -0.79,  0.65, -0.28,  0.09],
        'P': [-0.01, -1.58,  0.91,  0.68, -2.17, -0.66, -0.77, -0.01,  0.57, -0.47],
        'Q': [-0.58, -0.20, -1.40, -0.58, -1.62, -0.94,  0.01, -0.67, -0.07,  0.25],
        'R': [-1.96, -0.04, -0.12, -1.20,  0.35, -0.05,  0.10,  1.02,  0.22,  0.20],
        'S': [-0.92, -1.03, -1.38,  0.29, -1.62,  0.97,  0.24, -0.29, -1.41, -0.08],
        'T': [-0.36, -2.55,  1.04, -0.49, -1.56,  0.55,  0.64,  1.13, -0.47, -0.16],
        'V': [ 0.76, -2.23,  2.62, -0.52,  0.02, -0.88,  0.26,  0.77, -0.69, -0.29],
        'W': [ 0.48,  2.91,  1.15, -2.21,  1.76,  0.71, -0.07,  1.38,  0.75, -0.30],
        'Y': [ 0.16,  1.06,  0.49, -0.44, -0.14,  1.11, -0.24,  0.32, -0.34, -0.15],
    }

    FACTOR_NAMES = {
        'atchley': ['polarity', 'propensity_for_secondary_structures', 'molecular_volume',
                    'codon_diversity', 'electrostatic_charge'],
        'kidera': ['helix_bend_preference', 'side_chain_size', 'extended_structure_preference',
                   'hydrophobicity', 'double_bend_preference', 'partial_specific_volume',
                   'flat_extended_preference', 'occurrence_in_alpha_region', 'pK_C',
                   'surrounding_hydrophobicity'],
    }

    def __init__(self, factors: str, region_type: RegionType, scale_to_zero_mean: bool = False,
                 scale_to_unit_variance: bool = False, name: str = None):
        super().__init__(name=name)
        self.factors = factors
        self.region_type = region_type
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance
        self.factor_table = self.ATCHLEY_FACTORS if factors == 'atchley' else self.KIDERA_FACTORS
        self.scaler = None
        self._lookup, self._valid_mask = self._build_lookup_table()

    @staticmethod
    def build_object(dataset: Dataset, **params):
        location = AminoAcidPropertyEncoder.__name__
        ParameterValidator.assert_in_valid_list(params.get('factors'), ['atchley', 'kidera'],
                                                location, 'factors')
        ParameterValidator.assert_region_type(params, location)
        if 'scale_to_zero_mean' in params:
            ParameterValidator.assert_type_and_value(params['scale_to_zero_mean'], bool, location,
                                                     'scale_to_zero_mean')
        if 'scale_to_unit_variance' in params:
            ParameterValidator.assert_type_and_value(params['scale_to_unit_variance'], bool, location,
                                                     'scale_to_unit_variance')
        return AminoAcidPropertyEncoder(
            factors=params['factors'],
            region_type=RegionType[params['region_type'].upper()],
            scale_to_zero_mean=params.get('scale_to_zero_mean', False),
            scale_to_unit_variance=params.get('scale_to_unit_variance', False),
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
        loci = sorted(set(data.locus.tolist()))

        assert len(loci) == 2, (
            f"{self.__class__.__name__}: receptor dataset must contain exactly two chains, "
            f"but found: {loci}.")

        per_seq_embeddings = self._encode_sequence_set(data, seq_field)

        chain1, chain2 = {}, {}
        for i, (cell_id, locus) in enumerate(zip(data.cell_id.tolist(), data.locus.tolist())):
            (chain1 if locus == loci[0] else chain2)[cell_id] = per_seq_embeddings[i]

        assert set(chain1) == set(chain2), \
            f"{self.__class__.__name__}: some receptors are missing one of the two chains."

        receptor_ids = list(chain1.keys())
        examples = np.array([np.concatenate([chain1[cid], chain2[cid]]) for cid in receptor_ids])
        examples = self._scale_examples(examples, params)

        labels = (data.topandas().groupby('cell_id').first()[params.label_config.get_labels_by_name()]
                  .to_dict(orient='list') if params.encode_labels else None)

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
        return [f'{self.factors}_{name}' for name in self.FACTOR_NAMES[self.factors]]

    def _get_receptor_feature_names(self, loci: List[str]) -> List[str]:
        return [f'{locus}_{self.factors}_{name}'
                for locus in loci
                for name in self.FACTOR_NAMES[self.factors]]

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
