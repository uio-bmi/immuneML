import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.dsl.encoding_parsers.MatchedReferenceParser import MatchedReferenceParser
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMatchedReferenceParser(TestCase):
    def test_parse(self):

        data = {"Cell type": ["TCR_AB", "TCR_AB"],
                "Clonotype ID": [181, 182],
                "Chain: TRA (1)": ["LVGGVGGYNKLI", "LVGGVGGYNKLI"],
                "TRA - V gene (1)": ["TRAV4*01", "TRAV4*01"],
                "TRA - D gene (1)": [None, None],
                "TRA - J gene (1)": ["TRAJ4*01", "TRAJ4*01"],
                "Chain: TRA (2)": ["LVGGVGGYNKLI", "LVGGVGGYNKLI"],
                "TRA - V gene (2)": ["TRAV4*01", "TRAV4*01"],
                "TRA - D gene (2)": [None, None],
                "TRA - J gene (2)": ["TRAJ4*01", "TRAJ4*01"],
                "Chain: TRB (1)": ["LVGGVGGYNKLI", "LVGGVGGYNKLI"],
                "TRB - V gene (1)": ["TRAV4*01", "TRAV4*01"],
                "TRB - D gene (1)": [None, None],
                "TRB - J gene (1)": ["TRAJ4*01", "TRAJ4*01"],
                "Chain: TRB (2)": ["LVGGVGGYNKLI", "LVGGVGGYNKLI"],
                "TRB - V gene (2)": ["TRAV4*01", "TRAV4*01"],
                "TRB - D gene (2)": [None, None],
                "TRB - J gene (2)": ["TRAJ4*01", "TRAJ4*01"]}
        df = pd.DataFrame(data)

        path = EnvironmentSettings.root_path + "test/tmp/matchedrefparser/"
        PathBuilder.build(path)
        df.to_csv(path + "refs.csv", sep=";")

        specs = {
            "reference_sequences": {
                "path": path + "refs.csv",
                "format": "IRIS"  # or VDJdb
            },
            # "summary": "count",  # or percentage or clonotype_frequency (-> count sequences)
        }

        parsed, full_specs = MatchedReferenceParser.parse(specs)
        self.assertEqual(4, len(parsed["reference_sequences"]))
        self.assertEqual(2, parsed["max_distance"])
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in parsed["reference_sequences"]]))
        self.assertEqual(SequenceMatchingSummaryType.COUNT, parsed["summary"])

        specs = {
            "reference_sequences": {
                "path": path + "refs.csv",
                "format": "IRIS"  # or VDJdb
            },
            "summary": "percentage",  # or percentage or clonal_percentages (-> count sequences)
            "max_distance": 0
        }

        parsed, full_specs = MatchedReferenceParser.parse(specs)
        self.assertEqual(4, len(parsed["reference_sequences"]))
        self.assertEqual(0, parsed["max_distance"])
        self.assertTrue(all([isinstance(seq, ReceptorSequence) for seq in parsed["reference_sequences"]]))
        self.assertEqual(SequenceMatchingSummaryType.PERCENTAGE, parsed["summary"])

        specs["reference_sequences"]["format"] = 0
        self.assertRaises(AssertionError, MatchedReferenceParser.parse, specs)

        shutil.rmtree(path)
