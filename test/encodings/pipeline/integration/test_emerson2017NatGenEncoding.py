import shutil
from unittest import TestCase

import numpy as np

from source.analysis.criteria_matches.BooleanType import BooleanType
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.pipeline.PipelineEncoder import PipelineEncoder
from source.encodings.pipeline.steps.FisherExactFeatureAnnotation import FisherExactFeatureAnnotation
from source.encodings.pipeline.steps.PresentTotalFeatureTransformation import PresentTotalFeatureTransformation
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestEmerson2018NatGenEncoding(TestCase):

    def test_encode(self):

        path = EnvironmentSettings.root_path + "test/tmp/emerson2017natgenencoding/"
        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("diabetes", ["T1D", "FDR", "Ctl"])
        lc.add_label("aab", [0, 1, 2, 3])

        dataset_params = {
            "diabetes": np.array(["T1D", "Ctl", "FDR", "Ctl", "T1D", "Ctl", "FDR", "CTL"]),
            "aab": np.array([2, 1, 2, 1, 2, 1, 2, 1])
        }

        dataset_filenames, metadata = RepertoireBuilder.build(
            [
                ["AAA", "ATA", "ATA"],
                ["ATA", "TAA", "AAC"],
                ["AAA", "ATA", "ATA"],
                ["ATA", "TAA", "AAC"],
                ["AAA", "ATA", "ATA"],
                ["ATA", "TAA", "AAC"],
                ["AAA", "ATA", "ATA"],
                ["ASKLDFJD", "TAA", "AAC"]
            ],
            path,
            dataset_params
        )

        dataset = RepertoireDataset(
            filenames=dataset_filenames,
            params=dataset_params
        )

        fisher_exact_params = {
            "positive_criteria":
                {
                    "type": BooleanType.OR,
                    "operands": [
                        {
                            "type": OperationType.IN,
                            "value": {
                                "type": DataType.COLUMN,
                                "name": "diabetes"
                            },
                            "allowed_values": ["T1D"]
                        },
                        {
                            "type": BooleanType.AND,
                            "operands": [
                                {
                                    "type": OperationType.IN,
                                    "value": {
                                        "type": DataType.COLUMN,
                                        "name": "diabetes"
                                    },
                                    "allowed_values": ["FDR"]
                                },
                                {
                                    "type": OperationType.GREATER_THAN,
                                    "value": {
                                        "type": DataType.COLUMN,
                                        "name": "aab"
                                    },
                                    "threshold": 2
                                }
                            ]
                        }
                    ]
                },
        }

        transform_params = {
            "criteria": {
                "type": OperationType.LESS_THAN,
                "value": {
                    "type": DataType.COLUMN,
                    "name": "fisher_p.two_tail"
                },
                "threshold": 0.5
            }
        }

        kmer_freq_params = {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
            "reads": ReadsType.UNIQUE,
            "sequence_encoding": SequenceEncodingType.IDENTITY,
            "metadata_fields_to_include": []
        }

        params = EncoderParams(
                result_path=path,
                label_configuration=lc,
                batch_size=2,
                learn_model=True,
                filename="test.pickle",
                model={
                    "initial_encoder": KmerFrequencyEncoder.create_encoder(dataset),
                    "initial_encoder_params": kmer_freq_params,
                    "steps": [FisherExactFeatureAnnotation(**fisher_exact_params, filename="test.pickle", result_path=path), PresentTotalFeatureTransformation(**transform_params, filename="test.pickle", result_path=path)]
                }
            )

        d1 = PipelineEncoder.encode(
            dataset,
            params
        )

        params["learn_model"] = False
        params["filename"] = "test_2.pickle"

        dataset2 = RepertoireDataset(filenames=[dataset_filenames[num] for num in range(1, 4)])

        d2 = PipelineEncoder.encode(
            dataset2,
            params
        )

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(d1.encoded_data.examples.shape == (8, 2))
        self.assertTrue(isinstance(d2, RepertoireDataset))
        self.assertTrue(d2.encoded_data.examples.shape == (3, 2))
        self.assertTrue(np.array_equal(d1.encoded_data.examples[1:4, :].A, d2.encoded_data.examples.A))

        shutil.rmtree(path)
