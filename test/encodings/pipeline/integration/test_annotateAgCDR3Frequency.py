import os
import shutil
from unittest import TestCase

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.pipeline.PipelineEncoder import PipelineEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestAnnotateAgCDR3Frequency(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):

        root_path = EnvironmentSettings.root_path + "test/tmp/TestAnnotateAgCDR3Frequency/"

        lc = LabelConfiguration()
        lc.add_label("diabetes", ["T1D", "FDR", "Ctl"])
        lc.add_label("aab", [0, 1, 2, 3])

        dataset_params = {
            "diabetes": ["T1D", "Ctl", "FDR", "Ctl", "Ctl", "FDR", "CTL"],
            "aab": [2, 1, 2, 1, 1, 2, 1]
        }

        dataset = RepertoireDataset(
            repertoires=RepertoireBuilder.build(
                [
                    ["AAA", "ATA", "ATA"],
                    ["ATA", "TAA", "AAC"],
                    ["AAA", "ATA", "ATA"],
                    ["ATA", "TAA", "AAC"],
                    ["ATA", "TAA", "AAC"],
                    ["AAA", "ATA", "ATA"],
                    ["ASKLDFJD", "TAA", "AAC"]
                ],
                root_path,
                dataset_params
            )[0],
            params=dataset_params
        )

        reference_rep = """TRBV Gene	CDR3B AA Sequence	Antigen Protein	MHC Class									
VGENE1	CAAAF	A	MHC I									
VGENE2	CATAF	B	MHC II									
VGENE3	CASKLDFJDF	C	MHC II									
VGENE1	CASSIEGPTGELFF	D Transporter 8	MHC I"""

        reference_metadata = """filename,subject_id
reference_rep.tsv,rep1"""

        path = root_path + "/reference/"
        PathBuilder.build(path)
        with open(path + "reference_rep.tsv", "w") as file:
            file.writelines(reference_rep)
        with open(path + "metadata.csv", "w") as file:
            file.writelines(reference_metadata)

        reference_data_loader_params = {
            "result_path": path,
            "column_mapping": {
                "CDR3B AA Sequence": "sequence_aas",
                "TRBV Gene": "v_genes"
            },
            "columns_to_load": ["CDR3B AA Sequence", "TRBV Gene", "Antigen Protein", "MHC Class"],
            "metadata_file": path + "metadata.csv",
            "separator": "\t",
            "region_type": RegionType.CDR3.name,
            "region_definition": RegionDefinition.IMGT.name
        }

        kmer_freq_params = {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.IDENTITY.name,
            "metadata_fields_to_include": []
        }

        annotate_params = {
            "reference_sequence_path": path,
            "data_loader_name": "GenericImport",
            "data_loader_params": reference_data_loader_params,
            "sequence_matcher_params": {
                "max_distance": 0,
                "metadata_fields_to_match": [],
                "same_length": True
            },
            "annotation_prefix": "t1d_"
        }

        encoder_params = EncoderParams(
                result_path=path,
                label_config=lc,
                learn_model=True,
                model={}
            )

        encoder = PipelineEncoder.build_object(dataset, **{
                    "initial_encoder": KmerFrequencyEncoder.__name__[:-7],
                    "initial_encoder_params": kmer_freq_params,
                    "steps": [{'step1': {"type": "SequenceMatchFeatureAnnotation",
                               "params": {**annotate_params, **{"filename": "test.pickle", "result_path": path}}}}]
                })

        d1 = encoder.encode(
            dataset,
            encoder_params
        )

        n_matched = d1.encoded_data.feature_annotations["t1d_matched_MHC Class"].count()

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertEqual(n_matched, 3)

        shutil.rmtree(root_path)
