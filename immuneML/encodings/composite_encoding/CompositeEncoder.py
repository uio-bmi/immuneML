from typing import List

import numpy as np
import pandas as pd

from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class CompositeEncoder(DatasetEncoder):
    """
    This encoder allows to combine multiple different encodings together, for example, KmerFrequency encoder
    with VGeneEncoder. The parameters for the different encoders are passed as a list of dictionaries, where each
    dictionary contains the parameters for one encoder. The different encoders are applied sequentially and their
    results concatenated together.

    **Dataset type:**
    - SequenceDatasets
    - ReceptorDatasets
    - RepertoireDatasets

    .. note::

        To combine multiple encodings (e.g., GeneFrequency and KmerFrequency), keep in mind how the ML method will
        use the encoded data downstream. Currently, the recommended way to use CompositeEncoder is with
        :ref:`LogRegressionCustomPenalty`, where you can specify which features should not be penalized.

    **Specification arguments:**

    - encoders (list): A list of dictionaries, where each dictionary contains the parameters for one encoder.

    **YAML specification:**

    .. code-block:: yaml

        encodings:
            my_composite_encoding:
                Composite:
                    encoders:
                        - KmerFrequency:
                            k: 3
                        - GeneFrequency:
                            genes: [V]
                            normalization_type: relative_frequency
                            scale_to_unit_variance: true
                            scale_to_zero_mean: true

    """

    def __init__(self, encoders: List[DatasetEncoder], name: str = None):
        super().__init__(name=name)
        self.encoders = encoders

    @staticmethod
    def build_object(dataset: Dataset, **params):
        assert 'encoders' in params, "Parameter 'encoders' must be provided for CompositeEncoder."
        ParameterValidator.assert_all_type_and_value(params['encoders'], dict, "CompositeEncoder", 'encoders')
        name = params.get('name', 'composite')

        encoders = []
        for step, encoder_specs in enumerate(params['encoders']):
            cls_name = list(encoder_specs.keys())[0] + 'Encoder'
            encoder = ReflectionHandler.get_class_by_name(cls_name, 'encodings')
            default_params = DefaultParamsLoader.load('encodings', cls_name.replace('Encoder', ''))
            encoder_instance = encoder.build_object(dataset, **{**default_params, **encoder_specs[list(encoder_specs.keys())[0]]})
            encoder_instance.name = f"{name}_step_{step+1}_{encoder_instance.name or cls_name}"
            encoders.append(encoder_instance)

        return CompositeEncoder(encoders=encoders, name=name)

    def encode(self, dataset, params: EncoderParams) -> Dataset:
        examples, feature_names, feature_annotations, info = [], [], [], {}

        for encoder in self.encoders:
            encoded_dataset = encoder.encode(dataset, params)
            if examples is None:
                examples = [encoded_dataset.encoded_data.examples]
                feature_names = encoded_dataset.encoded_data.feature_names
                feature_annotations = encoded_dataset.encoded_data.feature_annotations
                feature_annotations['encoder'] = encoder.name
                feature_annotations.append(feature_annotations)
                info = {f'encoder_{encoder.name}': encoded_dataset.encoded_data.info}
            else:
                examples.append(encoded_dataset.encoded_data.examples)
                feature_names += encoded_dataset.encoded_data.feature_names
                feature_annotations.append(encoded_dataset.encoded_data.feature_annotations)
                info[f'encoder_{encoder.name}'] = encoded_dataset.encoded_data.info

        examples = NumpyHelper.concat_arrays_rowwise(examples, use_memmap=True)
        feature_annotations = pd.concat(feature_annotations, ignore_index=True)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=np.array(examples), feature_names=feature_names,
                                                   feature_annotations=feature_annotations, info=info,
                                                   labels=dataset.get_metadata(params.label_config.get_labels_by_name()))

        return encoded_dataset
