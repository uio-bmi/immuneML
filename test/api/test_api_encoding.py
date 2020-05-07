import csv
import glob
import os
import shutil
from unittest import TestCase

import numpy as np

from source.api.api_encoding import encode_dataset_by_kmer_freq
from source.caching.CacheType import CacheType
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestAPI(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_initial_dataset(self, data_path, repertoire_count):
        PathBuilder.build(data_path)

        for repertoire_index in range(1, repertoire_count + 1):
            with open(f"{data_path}rep_{repertoire_index}.tsv", "w") as file:
                writer = csv.DictWriter(file,
                                        delimiter="\t",
                                        fieldnames=["patient", "dilution", "cloneCount", "allVHitsWithScore",
                                                    "allJHitsWithScore", "nSeqCDR1", "nSeqCDR2", "nSeqCDR3", "minQualCDR3",
                                                    "aaSeqCDR1", "aaSeqCDR2", "aaSeqCDR3", "sampleID"])
                dicts = [{
                    "patient": "CD12",
                    "dilution": "108'",
                    "cloneCount": 3,
                    "allVHitsWithScore": "TRAV13-1*00(735)",
                    "allJHitsWithScore": "TRAJ15*00(243)",
                    "nSeqCDR1": "TGTGCAGCAA",
                    "nSeqCDR2": "TGTGCAGCAA",
                    "nSeqCDR3": "TGTGCAGCAA",
                    "minQualCDR3": 10,
                    "aaSeqCDR1": "CAASNQA",
                    "aaSeqCDR2": "CAASNQA",
                    "aaSeqCDR3": "CAASNQA",
                    "sampleID": "1"
                }, {
                    "patient": "CD12",
                    "dilution": "108'",
                    "cloneCount": 6,
                    "allVHitsWithScore": "TRAV19-1*00(735)",
                    "allJHitsWithScore": "TRAJ12*00(243)",
                    "nSeqCDR1": "CAATGTGA",
                    "nSeqCDR2": "CAATGTGA",
                    "nSeqCDR3": "CAATGTGA",
                    "minQualCDR3": 10,
                    "aaSeqCDR1": "CAASNTTA",
                    "aaSeqCDR2": "CAASNTTA",
                    "aaSeqCDR3": "CAASNTTA",
                    "sampleID": 1
                }, {
                    "patient": "CD12",
                    "dilution": "108'",
                    "cloneCount": 6,
                    "allVHitsWithScore": "TRAV19-1*00(735)",
                    "allJHitsWithScore": "TRAJ12*00(243)",
                    "nSeqCDR1": "CAATGTGA",
                    "nSeqCDR2": "CAATGTGA",
                    "nSeqCDR3": "CAATGTGA",
                    "minQualCDR3": 10,
                    "aaSeqCDR1": "CAASNTTA",
                    "aaSeqCDR2": "CAASNTTA",
                    "aaSeqCDR3": "CAASNTTA",
                    "sampleID": 1
                }, {
                    "patient": "CD12",
                    "dilution": "108'",
                    "cloneCount": 6,
                    "allVHitsWithScore": "TRAV19-1*00(735)",
                    "allJHitsWithScore": "TRAJ12*00(243)",
                    "nSeqCDR1": "CAATGTGA",
                    "nSeqCDR2": "CAATGTGA",
                    "nSeqCDR3": "CAATGTGA",
                    "minQualCDR3": 10,
                    "aaSeqCDR1": "CAASNTTA",
                    "aaSeqCDR2": "CAASNTTA",
                    "aaSeqCDR3": "CAASNTTA",
                    "sampleID": 1
                }, {
                    "patient": "CD12",
                    "dilution": "108'",
                    "cloneCount": 6,
                    "allVHitsWithScore": "TRAV19-1*00(735)",
                    "allJHitsWithScore": "TRAJ12*00(243)",
                    "nSeqCDR1": "CAATGTGA",
                    "nSeqCDR2": "CAATGTGA",
                    "nSeqCDR3": "CAATGTGA",
                    "minQualCDR3": 10,
                    "aaSeqCDR1": "CAASNTTA",
                    "aaSeqCDR2": "CAASNTTA",
                    "aaSeqCDR3": "CAASNTTA",
                    "sampleID": 1
                }]

                writer.writeheader()
                writer.writerows(dicts)

    def test_encode_dataset_by_kmer_freq(self):
        path = f"{EnvironmentSettings.tmp_test_path}testapi/"
        data_path = f"{path}data/"
        result_path = f"{path}result/"
        repertoire_count = 10

        self.create_initial_dataset(data_path, repertoire_count)

        encoded_dataset = encode_dataset_by_kmer_freq(path_to_dataset_directory=data_path, result_path=result_path)

        self.assertEqual(repertoire_count, len(glob.glob(f"{result_path}*.npy")))
        self.assertTrue(os.path.isfile(f"{result_path}csv_exported/design_matrix.csv"))
        self.assertTrue(os.path.isfile(f"{result_path}csv_exported/encoding_details.yaml"))
        self.assertTrue(os.path.isfile(f"{result_path}csv_exported/labels.csv"))
        self.assertEqual(repertoire_count, encoded_dataset.get_example_count())

        self.assertEqual(np.greater_equal(encoded_dataset.encoded_data.examples.todense(), 0).sum(),
                         np.less_equal(encoded_dataset.encoded_data.examples.todense(), 1).sum())
        self.assertEqual(repertoire_count*encoded_dataset.encoded_data.examples.shape[1],
                         np.greater_equal(encoded_dataset.encoded_data.examples.todense(), 0).sum())

        shutil.rmtree(path)
