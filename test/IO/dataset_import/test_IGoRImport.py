import shutil
from unittest import TestCase

from source.IO.dataset_import.IGoRImport import IGoRImport
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestIGoRImport(TestCase):

    def write_dummy_files(self, path):
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

        with open(path + "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path + "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        with open(path + "metadata.csv", "w") as file:
            file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def test_load_file_content(self):
        """Test dataset content with and without a header included in the input file"""
        path = EnvironmentSettings.root_path + "test/tmp/io_igor_load/"

        PathBuilder.build(path)
        self.write_dummy_files(path)
        dataset = IGoRImport.import_dataset({"result_path": path, "metadata_file": path + "metadata.csv",
                                             "columns_to_load": ["seq_index", "nt_CDR3", "anchors_found", "is_inframe"],
                                             "path": path, "batch_size": 4, "import_out_of_frame": False,
                                             "import_with_stop_codon": False,
                                             "separator": ",", "region_definition": "IMGT", "region_type": "CDR3",
                                             "column_mapping": {"nt_CDR3": "sequences",
                                                                "seq_index": "sequence_identifiers"}}, "igor_dataset")

        self.assertEqual(2, dataset.get_example_count())
        self.assertEqual(len(dataset.repertoires[0].sequences), 1)
        self.assertEqual(len(dataset.repertoires[1].sequences), 1)

        self.assertEqual(dataset.repertoires[0].sequences[0].amino_acid_sequence, "ARDRWSTPVLRYFDWWTPPYYYYMDV")
        shutil.rmtree(path)

    def test_load_with_stop_codon(self):
        path = EnvironmentSettings.root_path + "test/tmp/io_igor_load/"

        PathBuilder.build(path)
        self.write_dummy_files(path)

        dataset_stop_codons = IGoRImport.import_dataset({"result_path": path, "metadata_file": path + "metadata.csv",
                                             "columns_to_load": ["seq_index", "nt_CDR3", "anchors_found", "is_inframe"],
                                             "path": path, "batch_size": 4, "import_out_of_frame": False,
                                             "import_with_stop_codon": True,
                                             "separator": ",", "region_definition": "IMGT", "region_type": "CDR3",
                                             "column_mapping": {"nt_CDR3": "sequences",
                                                                "seq_index": "sequence_identifiers"}}, "igor_dataset")

        self.assertEqual(2, dataset_stop_codons.get_example_count())
        self.assertEqual(len(dataset_stop_codons.repertoires[0].sequences), 2)
        self.assertEqual(len(dataset_stop_codons.repertoires[1].sequences), 2)

        self.assertEqual(dataset_stop_codons.repertoires[0].sequences[0].amino_acid_sequence, "ARVNRHIVVVTAIMTG*NWFDP")

        shutil.rmtree(path)
