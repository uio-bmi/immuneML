from collections import Counter
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
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

    **Specification arguments:**

    - genes (list): List of genes to use for the encoding. Possible values are 'V', and 'J'. At least one gene must be
      specified.

    - normalization_type (str): Type of normalization to apply to the gene frequencies. Possible values are 'none',
      'binary', 'relative_frequency', 'max', 'l2'. Defaults to 'relative_frequency'.

    - scale_to_zero_mean (bool): Whether to scale the features to zero mean. Defaults to True.

    - scale_to_unit_variance (bool): Whether to scale the features to unit variance. Defaults to True.

    **YAML specification:**

    .. code-block:: yaml

        encodings:
            gene_frequency_encoding:
                GeneFrequency:
                    genes: [V, J]
                    normalization_type: relative_frequency
                    scale_to_unit_variance: true
                    scale_to_zero_mean: true

    """

    def __init__(self, genes: List[str], normalization_type: NormalizationType, scale_to_zero_mean: bool,
                 scale_to_unit_variance: bool, name: str = None):
        super().__init__(name=name)
        self.genes = genes
        self.normalization_type = normalization_type
        self.scale_to_zero_mean = scale_to_zero_mean
        self.scale_to_unit_variance = scale_to_unit_variance
        self.scaler = None
        self.feature_names = None

    @property
    def all_feature_names(self) -> List[str]:
        return [feature_name for gene in self.genes for feature_name in self.feature_names[gene]]

    @staticmethod
    def build_object(dataset: Dataset, **params):
        valid_keys = ['genes', 'normalization_type', 'scale_to_zero_mean', 'scale_to_unit_variance', 'name']
        ParameterValidator.assert_keys(params.keys(), valid_keys, "GeneFrequencyEncoder", "parameters", exclusive=False)
        ParameterValidator.assert_all_in_valid_list(params['genes'], ['V', 'J'], "GeneFrequencyEncoder", "genes")
        ParameterValidator.assert_type_and_value(params['scale_to_zero_mean'], bool, "GeneFrequencyEncoder",
                                                 "scale_to_zero_mean")
        ParameterValidator.assert_type_and_value(params['scale_to_unit_variance'], bool, "GeneFrequencyEncoder",
                                                 "scale_to_unit_variance")

        normalization_type = NormalizationType[params['normalization_type'].upper()]

        return GeneFrequencyEncoder(genes=params['genes'], normalization_type=normalization_type,
                                    scale_to_zero_mean=params['scale_to_zero_mean'],
                                    scale_to_unit_variance=params['scale_to_unit_variance'],
                                    name=params.get('name', 'gene_frequency'))

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        if isinstance(dataset, RepertoireDataset):
            return self._encode_repertoire_dataset(dataset, params)
        else:
            raise RuntimeError(f"{self.__class__.__name__}: {self.name}: invalid dataset type: {type(dataset)}.")

    def _encode_repertoire_dataset(self, dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        counters = {gene: [] for gene in self.genes}
        for rep in dataset.repertoires:
            for gene in self.genes:
                genes = [gene_call.split("*")[0] for gene_call in getattr(rep.data, f'{gene.lower()}_call').tolist()]
                counters[gene].append(Counter(genes))

        gene_dfs = {gene: pd.DataFrame(counters[gene]).fillna(0) for gene in self.genes}

        if params.learn_model:
            self.feature_names = {gene: df.columns.tolist() for gene, df in gene_dfs.items()}
        else:
            for gene in self.genes:
                gene_dfs[gene] = gene_dfs[gene].reindex(columns=self.feature_names[gene], fill_value=0)

        gene_dfs = {gene: FeatureScaler.normalize(df.values, self.normalization_type) for gene, df in gene_dfs.items()}
        examples = NumpyHelper.concat_arrays_rowwise([gene_dfs[gene] for gene in self.genes])

        if params.encode_labels:
            labels = dataset.get_metadata(params.label_config.get_labels_by_name(), return_df=False)
        else:
            labels = None

        return self._make_encoded_dataset(dataset, examples, labels, params)

    def _make_encoded_dataset(self, dataset, examples, labels, params: EncoderParams):

        examples = self._scale_examples(examples, params)

        encoded_data = EncodedData(examples=examples, labels=labels, example_ids=dataset.get_example_ids(),
                                   feature_names=self.all_feature_names, encoding=GeneFrequencyEncoder.__name__,
                                   info={'genes': self.genes},
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
