import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestOLGALoader(TestCase):

    def write_dummy_files(self, path, add_metadata):
        file1_content = """TGTGCCAGCAGTTTATCGCCGGGACTGGCCTACGAGCAGTACTTC	CASSLSPGLAYEQYF	TRBV27	TRBJ2-7
TGTGCCAGCAAAGTCAGAATTGCTGCAACTAATGAAAAACTGTTTTTT	CASKVRIAATNEKLFF	TRBV5-6	TRBJ1-4
TGCAGTGCCGACTCCAAGAACAGAGGAGCGGGGGGGGAGGCAAGCTCCTACGAGCAGTACTTC	CSADSKNRGAGGEASSYEQYF	TRBV20-1	TRBJ2-7"""
        file2_content = """TGTGCCAGCATCGGTGGCGGGACTAGTCTCTCCTACAATGAGCAGTTCTTC	CASIGGGTSLSYNEQFF	TRBV7-9	TRBJ2-1
TGTGCCAGTATCTGCGGATGTACTAGCACAGATACGCAGTATTTT	CASICGCTSTDTQYF	TRBV19	TRBJ2-3
TGTGCTAGTGGGAAAAATCGGGACTCTAGTGCAGGCCAAGAGACCCAGTACTTC	CASGKNRDSSAGQETQYF	TRBV12-5	TRBJ2-5"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def test_import_repertoire(self):
        """Test dataset content with and without a header included in the input file"""
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_olga_load_rep/")

        self.write_dummy_files(path, True)

        dataset = OLGAImport.import_dataset({"is_repertoire": True, "result_path": path, "metadata_file": path / "metadata.csv",
                                             "separator": "\t", "region_type": "IMGT_CDR3",
                                             "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                                             "import_illegal_characters": False, "columns_to_load": [0, 1, 2, 3],
                                             "column_mapping":
                                                 {0: "sequence",
                                                  1: "sequence_aa",
                                                  2: "v_call",
                                                  3: 'j_call'},
                                             "path": path, "number_of_processes": 4}, "olga_repertoire_dataset")

        self.assertEqual(2, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            self.assertEqual(3, len(rep.sequences))
            for sequence in rep.sequences:
                self.assertEqual(sequence.metadata.duplicate_count, 1)

            if index == 0:
                self.assertListEqual(["GCCAGCAGTTTATCGCCGGGACTGGCCTACGAGCAGTAC",
                                      "GCCAGCAAAGTCAGAATTGCTGCAACTAATGAAAAACTGTTT",
                                      "AGTGCCGACTCCAAGAACAGAGGAGCGGGGGGGGAGGCAAGCTCCTACGAGCAGTAC"],
                                     list(rep.get_attribute("sequence")))
                self.assertListEqual(["ASSLSPGLAYEQY",
                                      "ASKVRIAATNEKLF",
                                      "SADSKNRGAGGEASSYEQY"], list(rep.get_sequence_aas()))
                self.assertListEqual(["TRBV27", "TRBV5-6", "TRBV20-1"], list(rep.get_v_genes()))
                self.assertListEqual(["TRBJ2-7", "TRBJ1-4", "TRBJ2-7"], list(rep.get_j_genes()))
                self.assertListEqual([1, 1, 1], list(rep.get_counts()))
                self.assertListEqual([Chain.BETA, Chain.BETA, Chain.BETA], list(rep.get_chains()))

        shutil.rmtree(path)

    def test_import_sequences(self):
        """Test dataset content with and without a header included in the input file"""
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_olga_load_seq/")

        self.write_dummy_files(path, False)
        dataset = OLGAImport.import_dataset({"is_repertoire": False, "paired": False, "result_path": path,
                                             "columns_to_load": [1, 2, 3], "separator": "\t", "region_type": "IMGT_CDR3",
                                             "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                                             "import_illegal_characters": False, "column_mapping": {1: "sequence_aa", 2: "v_call", 3: "j_call"},
                                             "path": path, "number_of_processes": 4}, "olga_sequence_dataset")

        self.assertEqual(6, dataset.get_example_count())

        expected = ["ASSLSPGLAYEQY", "ASKVRIAATNEKLF", "SADSKNRGAGGEASSYEQY", "ASIGGGTSLSYNEQF", "ASICGCTSTDTQY", "ASGKNRDSSAGQETQY"]

        self.assertTrue(all(seq.amino_acid_sequence in expected for seq in dataset.get_data()))

        shutil.rmtree(path)
