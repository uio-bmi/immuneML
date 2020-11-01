import os
import shutil
from unittest import TestCase

from source.analysis.AxisType import AxisType
from source.analysis.criteria_matches.DataType import DataType
from source.analysis.criteria_matches.OperationType import OperationType
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.pipeline.PipelineEncoder import PipelineEncoder
from source.encodings.pipeline.steps.CriteriaBasedFilter import CriteriaBasedFilter
from source.encodings.pipeline.steps.PublicSequenceFeatureAnnotation import PublicSequenceFeatureAnnotation
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestTopPublicFeatureEncoding(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

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

        repertoires, metadata = RepertoireBuilder.build(
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
            repertoires=repertoires,
            params=dataset_params
        )

        encoder = PipelineEncoder.build_object(dataset, **{
                    "initial_encoder": KmerFrequencyEncoder.__name__[:-7],
                    "initial_encoder_params": {
                        "normalization_type": NormalizationType.NONE.name,
                        "reads": ReadsType.UNIQUE.name,
                        "sequence_encoding": SequenceEncodingType.IDENTITY.name,
                        "metadata_fields_to_include": [], "scale_to_unit_variance": False
                    },
                    "steps": [{"s1": {"type": PublicSequenceFeatureAnnotation.__name__,
                                      "params": {"result_path": path, "filename": "encoded_data.pickle"}}},
                              {"s2": {"type": CriteriaBasedFilter.__name__,
                                      "params": {"axis": AxisType.FEATURES,
                                                "criteria": {
                                                    "type": OperationType.TOP_N,
                                                    "number": 2,
                                                    "value": {
                                                        "type": DataType.COLUMN,
                                                        "name": "public_number_of_repertoires"
                                                    }
                                                }},
                        }}]
                })

        d1 = encoder.encode(
            dataset,
            EncoderParams(
                result_path=path,
                label_config=lc,
                learn_model=True,
                model={}
            )
        )

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(d1.encoded_data.examples.shape == (8, 2))
        self.assertTrue(d1.encoded_data.feature_annotations.shape == (2, 3))
        self.assertTrue(len(d1.encoded_data.feature_names) == 2)
