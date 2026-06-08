from typing import List, Optional

import numpy as np
import pandas as pd

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class CompositeEncoder(DatasetEncoder):
    """
    This encoder allows to combine multiple different encodings together, for example, KmerFrequency encoder
    with VGeneEncoder. The parameters for the different encoders are passed as a list of dictionaries, where each
    dictionary contains the parameters for one encoder. The different encoders are applied sequentially and their
    results concatenated together.

    Optionally, a dimensionality reduction method can be specified for each encoder via a ``dim_reduction`` key.
    When present, the reduction is fit on the training split and applied (transform-only) on subsequent splits.

    **Dataset type:**
    - SequenceDatasets
    - ReceptorDatasets
    - RepertoireDatasets

    .. note::

        To combine multiple encodings (e.g., GeneFrequency and KmerFrequency), keep in mind how the ML method will
        use the encoded data downstream. Currently, the recommended way to use CompositeEncoder is with
        :ref:`LogRegressionCustomPenalty`, where you can specify which features should not be penalized.

    **Specification arguments:**

    - encoders (list): A list of dictionaries, where each dictionary contains the parameters for one encoder and
      an optional ``dim_reduction`` key specifying a dimensionality reduction method to apply to that encoder's output.

    **YAML specification:**

    .. code-block:: yaml

        encodings:
            my_composite_encoding:
                Composite:
                    encoders:
                        - KmerFrequency:
                            k: 3
                          dim_reduction:
                            PCA:
                              n_components: 10
                        - GeneFrequency:
                            genes: [V]
                            normalization_type: relative_frequency
                            scale_to_unit_variance: true
                            scale_to_zero_mean: true

    """

    def __init__(self, encoders: List[DatasetEncoder], dim_red_methods: List[Optional[DimRedMethod]] = None,
                 name: str = None):
        super().__init__(name=name)
        self.encoders = encoders
        self.dim_red_methods = dim_red_methods if dim_red_methods is not None else [None] * len(encoders)

    @staticmethod
    def build_object(dataset: Dataset, **params):
        assert 'encoders' in params, "Parameter 'encoders' must be provided for CompositeEncoder."
        ParameterValidator.assert_all_type_and_value(params['encoders'], dict, "CompositeEncoder", 'encoders')
        name = params.get('name', 'composite')

        encoders = []
        dim_red_methods = []
        for step, encoder_specs in enumerate(params['encoders']):
            encoder_key = next(k for k in encoder_specs.keys() if k != 'dim_reduction')
            cls_name = encoder_key + 'Encoder'
            encoder = ReflectionHandler.get_class_by_name(cls_name, 'encodings')
            default_params = DefaultParamsLoader.load('encodings', cls_name.replace('Encoder', ''))
            encoder_instance = encoder.build_object(dataset, **{**default_params, **encoder_specs[encoder_key]})
            encoder_instance.name = f"{name}_step_{step+1}_{encoder_instance.name or cls_name}"
            encoders.append(encoder_instance)

            if 'dim_reduction' in encoder_specs:
                dim_red_spec = encoder_specs['dim_reduction']
                dim_red_cls_name = list(dim_red_spec.keys())[0]
                dim_red_cls = ReflectionHandler.get_class_by_name(dim_red_cls_name, 'ml_methods/dim_reduction/')
                dim_red_params = dim_red_spec[dim_red_cls_name] or {}
                dim_red_instance = dim_red_cls(**dim_red_params)
                dim_red_instance.name = f"{name}_step_{step+1}_dim_red"
                dim_red_methods.append(dim_red_instance)
            else:
                dim_red_methods.append(None)

        return CompositeEncoder(encoders=encoders, dim_red_methods=dim_red_methods, name=name)

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        examples, feature_names, feature_annotations, info = [], [], [], {}

        for encoder, dim_red_method in zip(self.encoders, self.dim_red_methods):
            encoded_dataset = encoder.encode(dataset, params)
            encoder_examples = encoded_dataset.encoded_data.examples

            if dim_red_method is not None:
                if params.learn_model:
                    encoder_examples = dim_red_method.fit_transform(design_matrix=encoder_examples)
                else:
                    encoder_examples = dim_red_method.transform(design_matrix=encoder_examples)
                encoder_feature_names = dim_red_method.get_dimension_names()
                tmp_feature_annotations = pd.DataFrame(index=range(len(encoder_feature_names)))
                tmp_feature_annotations['dim_reduction'] = type(dim_red_method).__name__
            else:
                encoder_feature_names = encoded_dataset.encoded_data.feature_names or []
                fa = encoded_dataset.encoded_data.feature_annotations
                n_features = len(encoder_feature_names)
                tmp_feature_annotations = fa.copy() if isinstance(fa, pd.DataFrame) and not fa.empty \
                    else pd.DataFrame(index=range(n_features))

            tmp_feature_annotations['encoder'] = type(encoder).__name__
            tmp_feature_annotations['encoder_name'] = encoder.name
            feature_annotations.append(tmp_feature_annotations)
            examples.append(encoder_examples)
            feature_names.extend(encoder_feature_names)
            info[f'encoder_{encoder.name}'] = encoded_dataset.encoded_data.info

        examples = NumpyHelper.concat_arrays_rowwise(examples, use_memmap=True)
        feature_annotations = pd.concat(feature_annotations, ignore_index=True)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=np.array(examples), feature_names=feature_names,
                                                   feature_annotations=feature_annotations, info=info,
                                                   labels=dataset.get_metadata(params.label_config.get_labels_by_name()),
                                                   encoding=type(self).__name__,
                                                   example_ids=dataset.get_example_ids())

        return encoded_dataset
