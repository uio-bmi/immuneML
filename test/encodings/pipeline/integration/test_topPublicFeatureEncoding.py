import shutil
from unittest import TestCase

from source.analysis.AxisType import AxisType
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.pipeline.PipelineEncoder import PipelineEncoder
from source.encodings.pipeline.steps.CriteriaBasedFilter import CriteriaBasedFilter
from source.encodings.pipeline.steps.PublicSequenceFeatureAnnotation import PublicSequenceFeatureAnnotation
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestTopPublicFeatureEncoding(TestCase):

    def test_encode(self):

        path = EnvironmentSettings.root_path + "test/tmp/emerson2017natgenencoding/"
        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("diabetes", ["T1D", "FDR", "Ctl"])
        lc.add_label("aab", [0, 1, 2, 3])

        dataset_params = {
            "diabetes": ["T1D", "Ctl", "FDR", "Ctl", "T1D", "Ctl", "FDR", "CTL"],
            "aab": [2, 1, 2, 1, 2, 1, 2, 1]
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

        dataset = Dataset(
            filenames=dataset_filenames,
            params=dataset_params
        )

        d1 = PipelineEncoder.encode(
            dataset,
            EncoderParams(
                result_path=path,
                label_configuration=lc,
                batch_size=2,
                learn_model=True,
                filename="encoded_data.pickle",
                model={
                    "initial_encoder": KmerFrequencyEncoder,
                    "initial_encoder_params": {
                        "normalization_type": NormalizationType.NONE,
                        "reads": ReadsType.UNIQUE,
                        "sequence_encoding": SequenceEncodingType.IDENTITY,
                        "metadata_fields_to_include": []
                    },
                    "steps": [PublicSequenceFeatureAnnotation(result_path=path, filename="encoded_data.pickle"), CriteriaBasedFilter(
                        **{"axis": AxisType.FEATURES,
                            "criteria": {
                                "type": OperationType.TOP_N,
                                "number": 2,
                                "value": {
                                    "type": DataType.COLUMN,
                                    "name": "public_number_of_repertoires"
                                }
                            },
                        })
                    ]
                }
            )
        )

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, Dataset))
        self.assertTrue(d1.encoded_data.repertoires.shape == (8, 2))
        self.assertTrue(d1.encoded_data.feature_annotations.shape == (2, 3))
        self.assertTrue(len(d1.encoded_data.feature_names) == 2)
