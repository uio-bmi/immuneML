import shutil
from unittest import TestCase

import pandas as pd

from source.IO.dataset_import.OLGALoader import OLGALoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestOLGALoader(TestCase):

    def write_dummy_files(self, path, header=False):
        if header:
            file1_content = '''sequences	sequence_aas	v_genes	j_genes
TGTGCCAGCAGTTTATCGCCGGGACTGGCCTACGAGCAGTACTTC	CASSLSPGLAYEQYF	TRBV27	TRBJ2-7
TGTGCCAGCAAAGTCAGAATTGCTGCAACTAATGAAAAACTGTTTTTT	CASKVRIAATNEKLFF	TRBV5-6	TRBJ1-4
TGCAGTGCCGACTCCAAGAACAGAGGAGCGGGGGGGGAGGCAAGCTCCTACGAGCAGTACTTC	CSADSKNRGAGGEASSYEQYF	TRBV20-1	TRBJ2-7'''
            file2_content = '''sequences	sequence_aas	v_genes	j_genes
TGTGCCAGCATCGGTGGCGGGACTAGTCTCTCCTACAATGAGCAGTTCTTC	CASIGGGTSLSYNEQFF	TRBV7-9	TRBJ2-1
TGTGCCAGTATCTGCGGATGTACTAGCACAGATACGCAGTATTTT	CASICGCTSTDTQYF	TRBV19	TRBJ2-3
TGTGCTAGTGGGAAAAATCGGGACTCTAGTGCAGGCCAAGAGACCCAGTACTTC	CASGKNRDSSAGQETQYF	TRBV12-5	TRBJ2-5'''

        else:
            file1_content = '''TGTGCCAGCAGTTTATCGCCGGGACTGGCCTACGAGCAGTACTTC	CASSLSPGLAYEQYF	TRBV27	TRBJ2-7
TGTGCCAGCAAAGTCAGAATTGCTGCAACTAATGAAAAACTGTTTTTT	CASKVRIAATNEKLFF	TRBV5-6	TRBJ1-4
TGCAGTGCCGACTCCAAGAACAGAGGAGCGGGGGGGGAGGCAAGCTCCTACGAGCAGTACTTC	CSADSKNRGAGGEASSYEQYF	TRBV20-1	TRBJ2-7'''
            file2_content = '''TGTGCCAGCATCGGTGGCGGGACTAGTCTCTCCTACAATGAGCAGTTCTTC	CASIGGGTSLSYNEQFF	TRBV7-9	TRBJ2-1
TGTGCCAGTATCTGCGGATGTACTAGCACAGATACGCAGTATTTT	CASICGCTSTDTQYF	TRBV19	TRBJ2-3
TGTGCTAGTGGGAAAAATCGGGACTCTAGTGCAGGCCAAGAGACCCAGTACTTC	CASGKNRDSSAGQETQYF	TRBV12-5	TRBJ2-5'''

        with open(path + "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path + "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        with open(path + "metadata.csv", "w") as file:
            file.writelines("""filename,donor
rep1.tsv,1
rep2.tsv,2""")

    def test_load_file_content(self):
        '''Test dataset content with and without a header included in the input file'''
        path = EnvironmentSettings.root_path + "test/tmp/io_olga/"

        # Test load function with and without headered data
        for use_header in (True, False):
            PathBuilder.build(path)
            self.write_dummy_files(path, use_header)
            dataset = OLGALoader().load(path=path, params={"result_path": path,
                                                           "metadata_file": path + "metadata.csv",
                                                           "has_header": use_header,
                                                           "col_names": ["sequences", "sequence_aas", "v_genes", "j_genes"]})

            self.check_dataset_content(dataset)
            shutil.rmtree(path)

    def check_dataset_content(self, dataset):
        '''Test that the dataset content is correct'''
        self.assertEqual(2, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            self.assertEqual(3, len(rep.sequences))
            for sequence in rep.sequences:
                self.assertEqual(sequence.metadata.count, 1)

            if index == 0:
                self.assertListEqual(["TGTGCCAGCAGTTTATCGCCGGGACTGGCCTACGAGCAGTACTTC",
                                      "TGTGCCAGCAAAGTCAGAATTGCTGCAACTAATGAAAAACTGTTTTTT",
                                      "TGCAGTGCCGACTCCAAGAACAGAGGAGCGGGGGGGGAGGCAAGCTCCTACGAGCAGTACTTC"],
                                     list(rep.get_attribute("sequences")))
                self.assertListEqual(["CASSLSPGLAYEQYF",
                                      "CASKVRIAATNEKLFF",
                                      "CSADSKNRGAGGEASSYEQYF"], list(rep.get_sequence_aas()))
                self.assertListEqual(["TRBV27", "TRBV5-6", "TRBV20-1"], list(rep.get_v_genes()))
                self.assertListEqual(["TRBJ2-7", "TRBJ1-4", "TRBJ2-7"], list(rep.get_j_genes()))

    def test_load_incorrect_params(self):
        '''An error should be thrown when loading with incorrect params'''
        path = EnvironmentSettings.root_path + "test/tmp/io_olga/"

        PathBuilder.build(path)
        self.write_dummy_files(path, False)

        self.assertRaises(IOError,
                           OLGALoader().load, path=path, params={"result_path": path,
                                                                   "metadata_file": path + "metadata.csv",
                                                                   "has_header": False,
                                                                   "col_names": []})

        self.assertRaises(IOError,
                          OLGALoader().load, path=path, params={"result_path": path,
                                                                "metadata_file": path + "metadata.csv",
                                                                "has_header": False,
                                                                "col_names": ["nonsense", "columns", "blabla", "bla"]})
        shutil.rmtree(path)

    def test_load_correct_params(self):
        '''No error should be thrown when minimal columns are present (sequences or sequence_aas)'''
        path = EnvironmentSettings.root_path + "test/tmp/io_olga/"

        PathBuilder.build(path)
        self.write_dummy_files(path, False)

        OLGALoader().load(path=path, params={"result_path": path,
                                             "metadata_file": path + "metadata.csv",
                                             "has_header": False,
                                             "col_names": ["sequences", "blablabla", "blabla", "bla"]})

        OLGALoader().load(path=path, params={"result_path": path,
                                             "metadata_file": path + "metadata.csv",
                                             "has_header": False,
                                             "col_names": ["blablabla", "sequence_aas", "blabla",
                                                           "bla"]})

        shutil.rmtree(path)





