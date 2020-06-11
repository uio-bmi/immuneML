import shutil
import unittest

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.api.galaxy.build_yaml_from_arguments import build_settings_specs, build_ml_methods_specs, get_sequence_enc_type, \
    build_encodings_specs
from source.api.galaxy.build_yaml_from_arguments import main as yamlbuilder_main
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class DummyArguments:
    pass


class MyTestCase(unittest.TestCase):
    def create_dummy_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["AA"], ["CC"]], path, labels={"label": ["lab1", "lab2"]})

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata)
        dataset.name = "my_dataset"
        PickleExporter.export(dataset, path)

        return f"{dataset.name}.iml_dataset"

    def test_main(self):
        path = PathBuilder.build(f"{EnvironmentSettings.tmp_test_path}args_to_yaml/")
        data_path = path + "/dummy_pickle_data/"

        iml_dataset_name = self.create_dummy_dataset(data_path)

        output_dir = f"{path}/output_dir"
        output_filename = "yaml_out.yaml"

        yamlbuilder_main(["-o", output_dir, "-f", output_filename,
                          "-d", f"{data_path}/{iml_dataset_name}", "-l", "label",
                          "-m", "SimpleLogisticRegression", "-t", "70",
                          "-c", "5", "-s", "subsequence", "subsequence",
                          "-r", "unique", "all",
                          "-g", "gapped", "ungapped",
                          "-p", "invariant", "positional",
                          "-kl", "2", "0", "-kr", "3", "0",
                          "-gi", "0", "0", "-ga", "4", "0",
                          "-k", "0", "3"])

        # Use ImmuneML parser to test whether the yaml file created here is still valid
        ImmuneMLParser.parse_yaml_file(f"{output_dir}/{output_filename}")

        shutil.rmtree(path)

    def test_get_sequence_enc_type(self):
        self.assertEqual(get_sequence_enc_type(sequence_type="complete", position_type=None, gap_type=None),
                         SequenceEncodingType.IDENTITY.value)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="positional", gap_type="gapped"),
                         SequenceEncodingType.IMGT_GAPPED_KMER.value)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="invariant", gap_type="gapped"),
                         SequenceEncodingType.GAPPED_KMER.value)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="positional", gap_type="ungapped"),
                         SequenceEncodingType.IMGT_CONTINUOUS_KMER.value)
        self.assertEqual(get_sequence_enc_type(sequence_type="subsequence", position_type="invariant", gap_type="ungapped"),
                         SequenceEncodingType.CONTINUOUS_KMER.value)

    def test_build_encodings_specs(self):
        args = DummyArguments
        args.sequence_type = ["complete", "subsequence", "subsequence", "subsequence", "subsequence"]
        args.position_type = [None, "positional", "invariant", "positional", "invariant"]
        args.gap_type = [None, "gapped", "gapped", "ungapped", "ungapped"]
        args.k = [None, None, None, 3, 4]
        args.k_left = [None, 2, 3, None, None]
        args.k_right = [None, 1, 5, None, None]
        args.min_gap = [None, 0, 5, None, None]
        args.max_gap = [None, 5, 10, None, None]
        args.reads = ["unique", "unique", "all", "all", "all"]

        result = build_encodings_specs(DummyArguments)

        correct = {"e1": {"KmerFrequency": {"sequence_encoding": "IdentitySequenceEncoder",
                                            "reads": "unique"}},
                   "e2": {"KmerFrequency": {"sequence_encoding": "IMGTGappedKmerEncoder",
                                            "reads": "unique",
                                            "k_left": 2,
                                            "k_right": 1,
                                            "min_gap": 0,
                                            "max_gap": 5}},
                   "e3": {"KmerFrequency": {"sequence_encoding": "GappedKmerSequenceEncoder",
                                            "reads": "all",
                                            "k_left": 3,
                                            "k_right": 5,
                                            "min_gap": 5,
                                            "max_gap": 10}},
                   "e4": {"KmerFrequency": {"sequence_encoding": "IMGTKmerSequenceEncoder",
                                            "reads": "all",
                                            "k": 3}},
                   "e5": {"KmerFrequency": {"sequence_encoding": "KmerSequenceEncoder",
                                            "reads": "all",
                                            "k": 4}}}

        self.assertDictEqual(result, correct)

    def test_build_ml_methods_specs(self):
        result = build_ml_methods_specs(["a", "b"])
        correct = {'ml1': 'a', 'ml2': 'b'}

        self.assertDictEqual(result, correct)

    def test_build_settings_specs(self):
        result = build_settings_specs(["a", "b"], ["c", "d"])
        correct = [{'encoding': 'a', 'ml_method': 'c'}, {'encoding': 'a', 'ml_method': 'd'}, {'encoding': 'b', 'ml_method': 'c'},
                   {'encoding': 'b', 'ml_method': 'd'}]

        self.assertListEqual(result, correct)


if __name__ == '__main__':
    unittest.main()
