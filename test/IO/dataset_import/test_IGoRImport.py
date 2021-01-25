import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.IGoRImport import IGoRImport
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestIGoRImport(TestCase):

    def write_dummy_files(self, path, add_metadata):
        file1_content = """seq_index,nt_CDR3,anchors_found,is_inframe
0,TGTGCGAGAGATCCTAGAAGCAGTGGCTGGAGATCAAAACCTACTGG,1,0
1,TGTGCGAGAGTTAATCGGCATATTGTGGTGGTGACTGCTATTATGACCGGGTAAAACTGGTTCGACCCCTGG,1,1
2,TGTGCGAGAGATAGGTGGTCAACCCCAGTATTACGATATTTTGACTGGTGGACCCCGCCCTACTACTACTACATGGACGTCTGG,1,1
3,TGTGCGAGAGGACCAAGCGGCCCTCAGAACGGTATGACTACTGG,1,0
4,GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTAAAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTGACTACTACATGAACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCATCCATTAGTAGTAGTAGTATCATATACTACGCAGACTCTGTGAAGGGCCGATTCACCATCTCCAGAGACAACGCCAAGAACTGACTGTATCTGCAAATGAACAGCCTGACAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGTCCCAGGGGGGATTACTATGATAGTAGTGGTTATTAGCCGGAACCCCACATGGACACACTCCTGGTGGTCCGCTTTTGATATCTGG,0,1
5,TGTGCGAGAGATCCGCGGTGTAGTGGTGGTAGCTGCTACTCCGACGAAGGCGCTGG,1,0"""
        file2_content = """seq_index,nt_CDR3,anchors_found,is_inframe
0,TGTGCGAGACGTGTCTAGGGAGGATATTGTAGTAGTACCAGCTGCTATGACGGGCGGTCCGGTAGTACTACTTTGACTACTGG,1,1
1,GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGGAAAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCGCTGGATTCACCTTCAGTGACTACTACATGAACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCATCCATTAGTAGTAGTAGTACCATATACTACGCAGACTCTGTGAAGGGCCGATTCACCATCTCCAGAGACAACGCCAAGAACTCACTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGATCCGGGAAGATTACTATGATAGTCGTGGTTATTACAAAGTTAAAGACTTTGACTACTGG,0,0
2,TTTGCGAGTCCATTCCTTCTCCAGTAGTACCAGCTGCTATCGGACTACTACTACTACATGGACGTCTGG,0,1
3,TGTGCACGTTTTTCTTGTAGTGGTGGTAGCTGCTACTGACTACTGG,1,0
4,TGTGCGAGAGGCTTCCATGGAACTACAGTAACTACGTTTGTAGGCTGTAGTACTACATGGACGTCTGG,1,1
5,TGTGCACACAGACCTTGGCCGGGTGAGTATTACGATTTTTGGAGTGGTTATTAGGCCTTTTTGACTACTGG,1,0"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def test_load_repertoire(self):
        """Test dataset content with and without a header included in the input file"""
        path = EnvironmentSettings.root_path / "test/tmp/io_igor_load/"

        PathBuilder.build(path)
        self.write_dummy_files(path, True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "igor")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["path"] = path
        params["metadata_file"] = path / "metadata.csv"

        dataset = IGoRImport.import_dataset(params, "igor_repertoire_dataset")


        self.assertEqual(2, dataset.get_example_count())
        self.assertEqual(len(dataset.repertoires[0].sequences), 1)
        self.assertEqual(len(dataset.repertoires[1].sequences), 1)

        self.assertEqual(dataset.repertoires[0].sequences[0].amino_acid_sequence, "ARDRWSTPVLRYFDWWTPPYYYYMDV")

        self.assertListEqual(list(dataset.repertoires[0].get_counts()), [1])
        self.assertEqual(dataset.repertoires[0].get_chains(), None)

        shutil.rmtree(path)

    def test_load_repertoire_with_stop_codon(self):
        path = EnvironmentSettings.root_path / "test/tmp/io_igor_load/"

        PathBuilder.build(path)
        self.write_dummy_files(path, True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "igor")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["path"] = path
        params["import_with_stop_codon"] = True
        params["metadata_file"] = path / "metadata.csv"

        dataset_stop_codons = IGoRImport.import_dataset(params, "igor_dataset_stop")


        self.assertEqual(2, dataset_stop_codons.get_example_count())
        self.assertEqual(len(dataset_stop_codons.repertoires[0].sequences), 2)
        self.assertEqual(len(dataset_stop_codons.repertoires[1].sequences), 2)

        self.assertEqual(dataset_stop_codons.repertoires[0].sequences[0].amino_acid_sequence, "ARVNRHIVVVTAIMTG*NWFDP")

        shutil.rmtree(path)

    def test_load_sequence_dataset(self):
        """Test dataset content with and without a header included in the input file"""
        path = EnvironmentSettings.root_path / "test/tmp/io_igor_load/"

        PathBuilder.build(path)
        self.write_dummy_files(path, False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "igor")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path
        params["import_with_stop_codon"] = True

        dataset = IGoRImport.import_dataset(params, "igor_seq_dataset")

        seqs = [sequence for sequence in dataset.get_data()]

        self.assertEqual(4, dataset.get_example_count())

        self.assertListEqual(sorted(["GCGAGACGTGTCTAGGGAGGATATTGTAGTAGTACCAGCTGCTATGACGGGCGGTCCGGTAGTACTACTTTGACTAC",
                              "GCGAGAGGCTTCCATGGAACTACAGTAACTACGTTTGTAGGCTGTAGTACTACATGGACGTC",
                              "GCGAGAGTTAATCGGCATATTGTGGTGGTGACTGCTATTATGACCGGGTAAAACTGGTTCGACCCC",
                              "GCGAGAGATAGGTGGTCAACCCCAGTATTACGATATTTTGACTGGTGGACCCCGCCCTACTACTACTACATGGACGTC"]),
                             sorted([seq.nucleotide_sequence for seq in seqs]))

        shutil.rmtree(path)