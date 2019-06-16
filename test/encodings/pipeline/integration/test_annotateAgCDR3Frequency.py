import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.pipeline.PipelineEncoder import PipelineEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.encodings.pipeline.steps.SequenceMatchFeatureAnnotation import SequenceMatchFeatureAnnotation


class TestAnnotateAgCDR3Frequency(TestCase):

    def test_encode(self):

        root_path = EnvironmentSettings.root_path + "test/tmp/TestAnnotateAgCDR3Frequency/"

        lc = LabelConfiguration()
        lc.add_label("diabetes", ["T1D", "FDR", "Ctl"])
        lc.add_label("aab", [0, 1, 2, 3])

        dataset_params = {
            "diabetes": ["T1D", "Ctl", "FDR", "Ctl", "T1D", "Ctl", "FDR", "CTL"],
            "aab": [2, 1, 2, 1, 2, 1, 2, 1]
        }

        dataset = Dataset(
            filenames=RepertoireBuilder.build(
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
                root_path,
                dataset_params
            ),
            params=dataset_params
        )

        reference_rep = """TRBV Gene	CDR3B AA Sequence	Antigen Protein	MHC Class									
VGENE1	CAAAF	A	MHC I									
VGENE2	CATAF	B	MHC II									
VGENE3	CASKLDFJDF	C	MHC II									
VGENE1	CASSIEGPTGELFF	D Transporter 8	MHC I"""

        reference_metadata = """filename
reference_rep.tsv"""

        path = root_path + "/reference/"
        PathBuilder.build(path)
        with open(path + "reference_rep.tsv", "w") as file:
            file.writelines(reference_rep)
        with open(path + "metadata.tsv", "w") as file:
            file.writelines(reference_metadata)

        reference_data_loader_params = {
            "result_path": path,
            "dataset_id": "t1d_verified",
            "extension": "tsv",
            "column_mapping": {
                "amino_acid": "CDR3B AA Sequence",
                "v_gene": "TRBV Gene"
            },
            "additional_columns": ["Antigen Protein", "MHC Class"],
            "strip_CF": True,
            "metadata_file": path + "metadata.tsv"
        }

        kmer_freq_params = {
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
            "reads": ReadsType.UNIQUE,
            "sequence_encoding_strategy": SequenceEncodingType.IDENTITY,
            "metadata_fields_to_include": []
        }

        annotate_params = {
            "reference_sequence_path": path,
            "data_loader_name": "GenericLoader",
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
                label_configuration=lc,
                batch_size=2,
                learn_model=True,
                filename="test.pickle",
                model={
                    "initial_encoder": KmerFrequencyEncoder(),
                    "initial_encoder_params": kmer_freq_params,
                    "steps": [SequenceMatchFeatureAnnotation(**annotate_params, filename="test.pickle", result_path=path)]
                }
            )

        d1 = PipelineEncoder.encode(
            dataset,
            encoder_params
        )

        n_matched = d1.encoded_data.feature_annotations["t1d_matched_MHC Class"].count()

        self.assertTrue(isinstance(d1, Dataset))
        self.assertEqual(n_matched, 3)

        shutil.rmtree(root_path)
