import glob
import os
import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.encoding_reports.MatchingSequenceDetails import MatchingSequenceDetails
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchingSequenceDetails(TestCase):
    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/encrepmatchingseq/"
        filenames = RepertoireBuilder.build([["AAA", "CCC"], ["AAC", "ASDA"], ["CCF", "ATC"]], path, {"default": [1, 0, 0]})
        ref_seqs = [ReceptorSequence("AAA", metadata=SequenceMetadata()),
                    ReceptorSequence("CCF", metadata=SequenceMetadata())]
        dataset = Dataset(filenames=filenames,
                          params={"default": [0, 1]},
                          encoded_data={
                              "repertoires": [[2], [1], [1]],
                              "labels": [[1, 0, 0]],
                              "label_names": ["default"],
                              "feature_names": ["percentage_of_sequences_matched"]
                          })

        report = MatchingSequenceDetails()

        report.generate_report({
            "dataset": dataset,
            "result_path": path + "result/",
            "reference_sequences": ref_seqs,
            "max_distance": 1
        })

        self.assertTrue(os.path.isfile(path + "result/matching_sequence_overview.tsv"))
        self.assertEqual(4, len([name for name in glob.glob(path + "result/*.tsv") if os.path.isfile(name)]))

        df = pd.read_csv(path + "result/matching_sequence_overview.tsv", sep="\t")
        self.assertTrue(all([key in df.keys() for key in ["patient", "chain", "matching_sequence_count", "repertoire_size", "max_levenshtein_distance"]]))
        self.assertEqual(3, df.shape[0])

        shutil.rmtree(path)
