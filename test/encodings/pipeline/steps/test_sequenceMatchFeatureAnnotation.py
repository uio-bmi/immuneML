import os
import shutil
from unittest import TestCase

import pandas as pd
from scipy import sparse

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.RegionType import RegionType
from source.data_model.repertoire.Repertoire import Repertoire
from source.encodings.pipeline.steps.SequenceMatchFeatureAnnotation import SequenceMatchFeatureAnnotation
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceMatchFeatureAnnotation(TestCase):

    # 5 features, 5 repertoires. Each repertoire has 3 labels. Each feature has 2 annotations.
    # feature_1 and feature_2 should match v_gene and sequence for 1 mismatch
    # feature 3 should match sequence for 1 mismatch
    # feature_4 should match sequence for 0 mismatches
    # feature 5 should match no sequences

    # test cases: 1: match v_gene and sequence, 1 mismatch: 2 matches
    #             2: match sequence, 1 mismatch: 4 matches
    #             3: match sequence, 0 mismatch: 1 match
    #             4: match v_gene and sequence, 0 mismatch: 0 matches

    path = EnvironmentSettings.root_path / "test/tmp/sequencematchfeatureannotationstep/"

    encoded_data = {
        'examples': sparse.rand(5, 5, density=0.2, format='csr'),
        'example_ids': ["A", "B", "C", "D", "E"],
        'labels': {
            "diabetes": [1, 0, 0, 1, 1],
        },
        'feature_names': ["VGENE1///AADAAA", "VGENE2///NNNNDN", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
        'feature_annotations': pd.DataFrame({
            "feature": ["VGENE1///AADAAA", "VGENE2///NNNNDN", "VGENE4///DDDDDE", "VGENE6///DDDDDD", "VGENE7///FFFFFF"],
            "sequence": ["AADAAA", "NNNNDN", "DDDDDE", "DDDDDD", "FFFFFF"],
            "v_gene": ["VGENE1", "VGENE2", "VGENE4", "VGENE6", "VGENE7"]
        })
    }

    dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data),
                                repertoires=[Repertoire(EnvironmentSettings.root_path / "test/tmp/sequencematchfeatureannotationstep/0.npy",
                                                        None, identifier) for identifier in encoded_data["example_ids"]])

    # there are only matches to the first 3 sequences in this reference
    reference_rep = """TRBV Gene	CDR3B AA Sequence	Antigen Protein	MHC Class									
VGENE1	CAAAAAAF	A	MHC I									
VGENE2	CNNNNNNF	B	MHC II									
VGENE3	CDDDDDDF	C	MHC II									
VGENE1	CASSIEGPTGELFF	D Transporter 8	MHC I"""

    reference_metadata = """filename,subject_id
reference_rep.tsv,rep1"""

    reference_data_loader_params = {
        "is_repertoire": True,
        "result_path": path,
        "column_mapping": {"CDR3B AA Sequence": "sequence_aas",
                           "TRBV Gene": "v_genes"},
        "columns_to_load": ["CDR3B AA Sequence", "TRBV Gene", "Antigen Protein", "MHC Class"],
        "region_type": RegionType.IMGT_CDR3.name,
        "metadata_file": path / "metadata.csv"
    }

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_transform_1(self):

        path = TestSequenceMatchFeatureAnnotation.path
        PathBuilder.build(path)
        with open(path / "reference_rep.tsv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_rep)
        with open(path / "metadata.csv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_metadata)

        step = SequenceMatchFeatureAnnotation(reference_sequence_path=path,
                                              data_loader_name="GenericImport",
                                              data_loader_params=TestSequenceMatchFeatureAnnotation.reference_data_loader_params,
                                              sequence_matcher_params={"max_distance": 1,
                                                                       "metadata_fields_to_match": ["v_gene"],
                                                                       "same_length": True},
                                              result_path=path,
                                              filename="encoded_dataset.iml_dataset",
                                              annotation_prefix="t1d_")

        dataset = step.fit_transform(TestSequenceMatchFeatureAnnotation.dataset)

        n_matched = dataset.encoded_data.feature_annotations["t1d_matched_sequence"].count()

        self.assertEqual(n_matched, 2)

        shutil.rmtree(path)

    def test_transform_2(self):

        path = TestSequenceMatchFeatureAnnotation.path
        PathBuilder.build(path)
        with open(path / "reference_rep.tsv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_rep)
        with open(path / "metadata.csv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_metadata)

        step = SequenceMatchFeatureAnnotation(reference_sequence_path=path,
                                              data_loader_name="GenericImport",
                                              data_loader_params=TestSequenceMatchFeatureAnnotation.reference_data_loader_params,
                                              sequence_matcher_params={"max_distance": 1,
                                                                       "metadata_fields_to_match": [],
                                                                       "same_length": True},
                                              result_path=path,
                                              filename="encoded_dataset.iml_dataset",
                                              annotation_prefix="t1d_")

        dataset = step.fit_transform(TestSequenceMatchFeatureAnnotation.dataset)

        n_matched = dataset.encoded_data.feature_annotations["t1d_matched_sequence"].count()

        self.assertEqual(n_matched, 4)

        shutil.rmtree(path)

    def test_transform_3(self):

        path = TestSequenceMatchFeatureAnnotation.path
        PathBuilder.build(path)
        with open(path / "reference_rep.tsv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_rep)
        with open(path / "metadata.csv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_metadata)

        step = SequenceMatchFeatureAnnotation(reference_sequence_path=path,
                                              data_loader_name="GenericImport",
                                              data_loader_params=TestSequenceMatchFeatureAnnotation.reference_data_loader_params,
                                              sequence_matcher_params={"max_distance": 0,
                                                                       "metadata_fields_to_match": [],
                                                                       "same_length": True},
                                              result_path=path,
                                              filename="encoded_dataset.iml_dataset",
                                              annotation_prefix="t1d_")

        dataset = step.fit_transform(TestSequenceMatchFeatureAnnotation.dataset)

        n_matched = dataset.encoded_data.feature_annotations["t1d_matched_sequence"].count()

        self.assertEqual(n_matched, 1)

        shutil.rmtree(path)

    def test_transform_4(self):

        path = TestSequenceMatchFeatureAnnotation.path
        PathBuilder.build(path)
        with open(path / "reference_rep.tsv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_rep)
        with open(path / "metadata.csv", "w") as file:
            file.writelines(TestSequenceMatchFeatureAnnotation.reference_metadata)

        step = SequenceMatchFeatureAnnotation(reference_sequence_path=path,
                                              data_loader_name="GenericImport",
                                              data_loader_params=TestSequenceMatchFeatureAnnotation.reference_data_loader_params,
                                              sequence_matcher_params={"max_distance": 0,
                                                                       "metadata_fields_to_match": ["v_gene"],
                                                                       "same_length": True},
                                              result_path=path,
                                              filename="encoded_dataset.iml_dataset",
                                              annotation_prefix="t1d_")

        dataset = step.fit_transform(TestSequenceMatchFeatureAnnotation.dataset)

        n_matched = dataset.encoded_data.feature_annotations["t1d_matched_sequence"].count()

        self.assertEqual(n_matched, 0)

        shutil.rmtree(path)
