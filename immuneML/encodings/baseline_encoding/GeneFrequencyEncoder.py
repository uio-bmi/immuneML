from collections import Counter
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.ParameterValidator import ParameterValidator


class GeneFrequencyEncoder(DatasetEncoder):
    """
    GeneFrequencyEncoder represents a repertoire by the frequency of V and/or J genes used.

    **Dataset type:**
    - RepertoireDatasets
    - ReceptorDatasets
    - SequenceDatasets

    **Specification arguments:**

    - genes (list): List of genes to use for the encoding. Possible values are 'V', and 'J'. At least one gene must be
      specified.

    - normalization_type (str): Type of normalization to apply to the gene frequencies. Possible values are 'none',
      'binary', 'relative_frequency', 'max', 'l2'. Defaults to 'relative_frequency'. For SequenceDatasets and
      ReceptorDatasets, the gene frequencies are binary (1 if the gene is present in any chain, 0 otherwise)
      regardless of the normalization_type specified.

    - encoding_type (str): How to encode gene presence for SequenceDatasets and ReceptorDatasets. Ignored for
      RepertoireDatasets. Possible values are 'dummy' and 'one_hot'. Defaults to 'dummy'.

      - 'dummy': Creates k−1 binary columns per gene segment and locus (V or J), dropping the most frequent gene as the
        reference. Since each receptor or sequence carries exactly one gene per segment, including all k binary
        columns would create perfect multicollinearity with the intercept (the columns sum to 1). Dropping the most
        frequent gene resolves this and is the standard approach when fitting a model with an intercept. The reference
        gene for each segment is stored in the encoded data info and reported in the results. Use this when fitting
        logistic regression with an intercept, or when combining with other encodings.

      - 'one_hot': Creates k binary columns, one per observed gene. Avoids multicollinearity only when the downstream
        model is fitted without an intercept (fit_intercept=False in sklearn). In that case, each coefficient directly
        represents a gene's absolute log-odds contribution rather than an effect relative to an arbitrary baseline.

    - scale_to_zero_mean (bool): Whether to scale the features to zero mean. Defaults to True.

    - scale_to_unit_variance (bool): Whether to scale the features to unit variance. Defaults to True.

    **YAML specification:**

    .. code-block:: yaml

        encodings:
            gene_frequency_encoding:
                GeneFrequency:
                    genes: [V, J]
                    normalization_type: relative_frequency
                    encoding_type: dummy
                    scale_to_unit_variance: true
                    scale_to_zero_mean: true

    """

    VALID_ENCODING_TYPES = ['dummy', 'one_hot']

    def __init__(self, genes: List[str], normalization_type: NormalizationType, scale_to_zero_mean: bool,
                 scale_to_unit_variance: bool, encoding_type: str = 'dummy', name: str = None):
        super().__init__(name=name)
        self.genes = genes
        self.normalization_type = normalization_type
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance
        self.encoding_type = encoding_type
        self.scaler = None
        self.feature_names = None
        self.reference_genes = {}

    @property
    def all_feature_names(self) -> List[str]:
        if self.feature_names is None:
            return []
        return [name for gene in self.genes for name in self.feature_names[gene]]

    @staticmethod
    def build_object(dataset: Dataset, **params):
        valid_keys = ['genes', 'normalization_type', 'scale_to_zero_mean', 'scale_to_unit_variance',
                      'encoding_type', 'name']
        ParameterValidator.assert_keys(params.keys(), valid_keys, "GeneFrequencyEncoder", "parameters", exclusive=False)
        ParameterValidator.assert_all_in_valid_list(params['genes'], ['V', 'J'], "GeneFrequencyEncoder", "genes")
        ParameterValidator.assert_type_and_value(params['scale_to_zero_mean'], bool, "GeneFrequencyEncoder",
                                                 "scale_to_zero_mean")
        ParameterValidator.assert_type_and_value(params['scale_to_unit_variance'], bool, "GeneFrequencyEncoder",
                                                 "scale_to_unit_variance")

        encoding_type = params.get('encoding_type', 'dummy')
        if encoding_type not in GeneFrequencyEncoder.VALID_ENCODING_TYPES:
            raise ValueError(f"GeneFrequencyEncoder: encoding_type must be one of "
                             f"{GeneFrequencyEncoder.VALID_ENCODING_TYPES}, got '{encoding_type}'.")

        normalization_type = NormalizationType[params['normalization_type'].upper()]

        return GeneFrequencyEncoder(genes=params['genes'], normalization_type=normalization_type,
                                    scale_to_zero_mean=params['scale_to_zero_mean'],
                                    scale_to_unit_variance=params['scale_to_unit_variance'],
                                    encoding_type=encoding_type,
                                    name=params.get('name', 'gene_frequency'))

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        if isinstance(dataset, RepertoireDataset):
            return self._encode_repertoire_dataset(dataset, params)
        elif isinstance(dataset, ReceptorDataset):
            return self._encode_receptor_dataset(dataset, params)
        elif isinstance(dataset, SequenceDataset):
            return self._encode_sequence_dataset(dataset, params)
        else:
            raise RuntimeError(f"{self.__class__.__name__}: {self.name}: invalid dataset type: {type(dataset)}.")

    def _drop_reference_gene(self, dummies: pd.DataFrame, segment: str, locus: str) -> pd.DataFrame:
        """Drop the most frequent gene as the reference for the given segment+locus (e.g. V_TRA, J_TRB)."""
        reference = dummies.sum().idxmax()
        self.reference_genes[f'{segment}_{locus}'] = reference
        return dummies.drop(columns=[reference])

    def _dummies_per_locus(self, df: pd.DataFrame, col: str, dtype) -> dict:
        """Return a dict of {locus -> dummies DataFrame} using the locus column in df."""
        locus_dfs = {}
        for locus, locus_df in df.groupby('locus'):
            gene_values = locus_df[col].apply(lambda x: x.split('*')[0] if x else x)
            locus_dummies = pd.get_dummies(gene_values, dtype=dtype)
            locus_dummies.index = locus_df.index
            locus_dfs[locus] = locus_dummies
        return locus_dfs

    def _encode_receptor_dataset(self, dataset: ReceptorDataset, params: EncoderParams) -> ReceptorDataset:
        df = dataset.data.topandas()

        gene_dfs = {}
        for segment in self.genes:
            col = f'{segment.lower()}_call'
            locus_dfs = {}

            for locus, locus_df in df.groupby('locus'):
                gene_values = locus_df[col].apply(lambda x: x.split('*')[0] if x else x)
                locus_dummies = pd.get_dummies(gene_values, dtype=float)
                locus_dummies['cell_id'] = locus_df['cell_id'].values
                locus_dummies = locus_dummies.groupby('cell_id').max()  # index = cell_id
                if params.learn_model and self.encoding_type == 'dummy':
                    locus_dummies = self._drop_reference_gene(locus_dummies, segment, locus)
                locus_dfs[locus] = locus_dummies

            # join loci side-by-side on cell_id index; reorder to match get_example_ids() order, then drop index
            combined = pd.concat(locus_dfs.values(), axis=1).fillna(0)
            combined = combined.loc[dataset.get_example_ids()].reset_index(drop=True)
            gene_dfs[segment] = combined

        if params.learn_model:
            self.feature_names = {segment: gene_df.columns.tolist() for segment, gene_df in gene_dfs.items()}
        else:
            for segment in self.genes:
                # genes not seen during training are ignored (dropped);
                # training genes absent here become 0; reference gene is absent by design
                gene_dfs[segment] = gene_dfs[segment].reindex(columns=self.feature_names[segment], fill_value=0)

        examples = NumpyHelper.concat_arrays_rowwise([gene_dfs[segment].values for segment in self.genes])

        labels = dataset.get_metadata(params.label_config.get_labels_by_name(), return_df=False) \
            if params.encode_labels else None

        encoded_data = EncodedData(examples=examples, labels=labels, example_ids=dataset.get_example_ids(),
                                   feature_names=self.all_feature_names, encoding=GeneFrequencyEncoder.__name__,
                                   info={'genes': self.genes, 'encoding_type': self.encoding_type,
                                         'reference_genes': self.reference_genes},
                                   feature_annotations=pd.DataFrame({"feature": self.all_feature_names}))
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data
        return encoded_dataset

    def _encode_sequence_dataset(self, dataset: SequenceDataset, params: EncoderParams) -> SequenceDataset:
        df = dataset.data.topandas()

        gene_dfs = {}
        for segment in self.genes:
            col = f'{segment.lower()}_call'
            locus_dfs = self._dummies_per_locus(df, col, dtype=int)

            for locus, locus_dummies in locus_dfs.items():
                if params.learn_model and self.encoding_type == 'dummy':
                    locus_dummies = self._drop_reference_gene(locus_dummies, segment, locus)
                locus_dfs[locus] = locus_dummies

            # stack loci row-wise (each sequence belongs to one locus), fill other-locus columns with 0
            combined = pd.concat(locus_dfs.values(), axis=0).fillna(0).sort_index().reset_index(drop=True)
            gene_dfs[segment] = combined

        if params.learn_model:
            self.feature_names = {segment: gene_df.columns.tolist() for segment, gene_df in gene_dfs.items()}
        else:
            for segment in self.genes:
                gene_dfs[segment] = gene_dfs[segment].reindex(columns=self.feature_names[segment], fill_value=0)

        examples = NumpyHelper.concat_arrays_rowwise([gene_dfs[segment].values for segment in self.genes])

        labels = dataset.get_metadata(params.label_config.get_labels_by_name(), return_df=False) \
            if params.encode_labels else None

        encoded_data = EncodedData(examples=examples, labels=labels, example_ids=dataset.get_example_ids(),
                                   feature_names=self.all_feature_names, encoding=GeneFrequencyEncoder.__name__,
                                   info={'genes': self.genes, 'encoding_type': self.encoding_type,
                                         'reference_genes': self.reference_genes},
                                   feature_annotations=pd.DataFrame({"feature": self.all_feature_names}))
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data
        return encoded_dataset

    def _encode_repertoire_dataset(self, dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        counters = {segment: [] for segment in self.genes}
        for rep in dataset.repertoires:
            for segment in self.genes:
                genes = [gene_call.split("*")[0] for gene_call in
                         getattr(rep.data, f'{segment.lower()}_call').tolist()]
                counters[segment].append(Counter(genes))

        gene_dfs = {segment: pd.DataFrame(counters[segment]).fillna(0) for segment in self.genes}

        if params.learn_model:
            self.feature_names = {segment: df.columns.tolist() for segment, df in gene_dfs.items()}
        else:
            for segment in self.genes:
                gene_dfs[segment] = gene_dfs[segment].reindex(columns=self.feature_names[segment], fill_value=0)

        gene_dfs = {segment: FeatureScaler.normalize(df.values, self.normalization_type)
                    for segment, df in gene_dfs.items()}
        examples = NumpyHelper.concat_arrays_rowwise([gene_dfs[segment] for segment in self.genes])

        if params.encode_labels:
            labels = dataset.get_metadata(params.label_config.get_labels_by_name(), return_df=False)
        else:
            labels = None

        return self._make_encoded_dataset(dataset, examples, labels, params)

    def _make_encoded_dataset(self, dataset, examples, labels, params: EncoderParams):

        examples = self._scale_examples(examples, params)

        encoded_data = EncodedData(examples=examples, labels=labels, example_ids=dataset.get_example_ids(),
                                   feature_names=self.all_feature_names, encoding=GeneFrequencyEncoder.__name__,
                                   info={'genes': self.genes, 'encoding_type': self.encoding_type,
                                         'reference_genes': self.reference_genes},
                                   feature_annotations=pd.DataFrame({"feature": self.all_feature_names}))

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data
        return encoded_dataset

    def _scale_examples(self, examples, params):
        if params.learn_model:
            self.scaler = StandardScaler(with_mean=self.scale_to_zero_mean, with_std=self.scale_to_unit_variance)
            examples = FeatureScaler.standard_scale_fit(self.scaler, examples, with_mean=self.scale_to_zero_mean)
        else:
            examples = FeatureScaler.standard_scale(self.scaler, examples, with_mean=self.scale_to_zero_mean)

        return examples