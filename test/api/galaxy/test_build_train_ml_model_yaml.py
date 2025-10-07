import os
import shutil
import unittest

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.api.galaxy.build_train_ml_model_yaml import build_settings_specs, build_ml_methods_specs, \
    get_sequence_enc_type, \
    build_encodings_specs, build_labels
from immuneML.api.galaxy.build_train_ml_model_yaml import main as yamlbuilder_main
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class DummyArguments:
    pass


class MyTestCase(unittest.TestCase):
    def create_dummy_dataset(self, path):
        dataset = RepertoireBuilder.build_dataset([["AA"], ["CC"]], path, labels={"label1": ["val1", "val2"], "label2": ["val1", "val2"]}, name="dataset")

        # dataset.name = "dataset"
        AIRRExporter.export(dataset, path)

        return f"dataset.yaml"

    def test_main(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "args_to_yaml")
        data_path = path / "dummy_pickle_data"

        iml_dataset_name = self.create_dummy_dataset(data_path)

        output_dir = path / "output_dir"
        output_filename = "yaml_out.yaml"

        old_wd = os.getcwd()

        try:
            os.chdir(data_path)

            yamlbuilder_main(["-o", str(output_dir), "-f", output_filename,
                              "-l", "label1,label2",
                              "-m", "LogisticRegression", "-t", "70",
                              "-c", "5", "-s", "subsequence", "subsequence",
                              "-r", "unique", "all",
                              "-g", "gapped", "ungapped",
                              "-p", "invariant", "positional",
                              "-kl", "2", "0", "-kr", "3", "0",
                              "-gi", "0", "0", "-ga", "4", "0",
                              "-k", "0", "3"])

            # Use ImmuneML parser to test whether the yaml file created here is still valid
            ImmuneMLParser.parse_yaml_file(output_dir / output_filename, path / "result_path")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)

    def test_get_sequence_enc_type(self):
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="positional", gap_type="gapped"),
                         SequenceEncodingType.IMGT_GAPPED_KMER.name)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="invariant", gap_type="gapped"),
                         SequenceEncodingType.GAPPED_KMER.name)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="positional", gap_type="ungapped"),
                         SequenceEncodingType.IMGT_CONTINUOUS_KMER.name)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="invariant", gap_type="ungapped"),
                         SequenceEncodingType.CONTINUOUS_KMER.name)

    def test_build_encodings_specs(self):
        args = DummyArguments
        args.sequence_type = ["subsequence", "subsequence", "subsequence", "subsequence"]
        args.position_type = ["positional", "invariant", "positional", "invariant"]
        args.gap_type = ["gapped", "gapped", "ungapped", "ungapped"]
        args.k = [None, None, 3, 4]
        args.k_left = [2, 3, None, None]
        args.k_right = [1, 5, None, None]
        args.min_gap = [0, 5, None, None]
        args.max_gap = [5, 10, None, None]
        args.reads = ["unique", "all", "all", "all"]

        result = build_encodings_specs(DummyArguments)

        correct = {"encoding_1": {"KmerFrequency": {"sequence_encoding": "IMGT_GAPPED_KMER",
                                                    "reads": "unique",
                                                    "k_left": 2,
                                                    "k_right": 1,
                                                    "min_gap": 0,
                                                    "max_gap": 5}},
                   "encoding_2": {"KmerFrequency": {"sequence_encoding": "GAPPED_KMER",
                                                    "reads": "all",
                                                    "k_left": 3,
                                                    "k_right": 5,
                                                    "min_gap": 5,
                                                    "max_gap": 10}},
                   "encoding_3": {"KmerFrequency": {"sequence_encoding": "IMGT_CONTINUOUS_KMER",
                                                    "reads": "all",
                                                    "k": 3}},
                   "encoding_4": {"KmerFrequency": {"sequence_encoding": "CONTINUOUS_KMER",
                                                    "reads": "all",
                                                    "k": 4}}}

        self.assertDictEqual(result, correct)

    def test_build_ml_methods_specs(self):
        args = DummyArguments
        args.ml_methods = ["a", "b"]
        args.neighbors = 0
        result = build_ml_methods_specs(args)
        correct = {'a': 'a', 'b': 'b'}

        self.assertDictEqual(result, correct)

    def test_build_settings_specs(self):
        result = build_settings_specs(["a", "b"], ["c", "d"])
        correct = [{'encoding': 'a', 'ml_method': 'c'}, {'encoding': 'a', 'ml_method': 'd'}, {'encoding': 'b', 'ml_method': 'c'},
                   {'encoding': 'b', 'ml_method': 'd'}]

        self.assertListEqual(result, correct)

    def test_build_labels(self):
        result = build_labels("'cmv_status, t1d_status\t,cd_status\"'")
        correct = ["cmv_status", "t1d_status", "cd_status"]

        self.assertListEqual(result, correct)


if __name__ == '__main__':
    unittest.main()
