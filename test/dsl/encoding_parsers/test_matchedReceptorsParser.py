import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.dsl.encoding_parsers.MatchedReceptorsParser import MatchedReceptorsParser
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMatchedReceptorsParser(TestCase):
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

        path = EnvironmentSettings.root_path + "test/tmp/matched_receptors_parser/"
        PathBuilder.build(path)
        df.to_csv(path + "refs.csv", sep=";")


        specs = {
            "reference_sequences": {
                "path": path + "refs.csv",
                "format": "IRIS",  # or VDJdb
                "paired": True
            },
        }

        parsed, full_specs = MatchedReceptorsParser.parse(specs)

        self.assertEqual(2, len(parsed["reference_sequences"]))
        self.assertTrue(all([isinstance(seq, TCABReceptor) for seq in parsed["reference_sequences"]]))
        self.assertEqual(True, parsed["one_file_per_donor"])
        self.assertEqual(True, full_specs["reference_sequences"]["paired"])

        specs["reference_sequences"]["format"] = 0
        self.assertRaises(AssertionError, MatchedReceptorsParser.parse, specs)


        # This is not allowed to work with not-paired data, assert that AssertionError should occur
        specs = {
            "reference_sequences": {
                "path": path + "refs.csv",
                "format": "IRIS",  # or VDJdb
                "paired": False
            },
        }

        self.assertRaises(AssertionError, MatchedReceptorsParser.parse, params=specs)

        shutil.rmtree(path)