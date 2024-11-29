import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.TenxGenomicsImport import TenxGenomicsImport
from immuneML.data_model.SequenceParams import Chain
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder



#     def create_dumy_dataset(self, path, add_metadata):
#         file1_content = """clonotype_id,consensus_id,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis
# clonotype100,clonotype100_consensus_1,843,TRA,TRAV9-2,,TRAJ52,TRAC,True,False,CVLVTGANTGKLTF,TGTGTTTTGGTCACTGGAGCTAACACTGGAAAGCTCACGTTT,8566,4
# clonotype100,clonotype100_consensus_2,685,TRB,TRBV20,,TRBJ2-7,TRBC1,True,False,CGARGQNYEQYF,TGTGGTGCTCGGGGACAAAACTATGAACAGTACTTC,74572,29
# clonotype101,clonotype101_consensus_1,620,TRA,TRAV12D-3,,TRAJ12,TRAC,True,True,CALSGTGGYKVVF,TGTGCTCTGAGTGGGACTGGAGGCTATAAAGTGGTCTTT,3396,2
# clonotype101,clonotype101_consensus_2,759,TRB,TRBV3,TRBD1,TRBJ1-1,TRBC1,True,True,CASSLYGGPEVFF,TGTGCCAGCAGCTTATATGGGGGCCCAGAAGTCTTCTTT,18133,4"""
#
#         file2_content = """clonotype_id,consensus_id,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis
# clonotype102,clonotype102_consensus_1,675,TRA,TRAV14N-1,None,TRAJ5,TRAC,True,True,CAAKGTQVVGQLTF,TGTGCAGCAAAGGGGACACAGGTTGTGGGGCAGCTCACTTTC,23380,4
# clonotype103,clonotype103_consensus_1,572,TRA,TRAV13D-2,None,TRAJ37,TRAC,True,True,CAIVGNTGKLIF,TGTGCTATAGTAGGCAATACCGGAAAACTCATCTTT,23899,13
# clonotype103,clonotype103_consensus_2,753,TRB,TRBV3,None,TRBJ1-2,TRBC1,True,True,CASSFATNSDYTF,TGTGCCAGCAGCTTCGCAACAAACTCCGACTACACCTTC,52713,28
# clonotype104,clonotype104_consensus_1,680,TRA,TRAV3D-3,None,TRAJ31,TRAC,True,True,CAVSANSNNRIFF,TGCGCAGTCAGTGCGAATAGCAATAACAGAATCTTCTTT,31289,6"""

class TestTenxGenomicsImport(TestCase):

    def create_dumy_dataset(self, path, add_metadata):
        file1_content = """barcode,is_cell,contig_id,high_confidence,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis,raw_clonotype_id,raw_consensus_id
AAACCTGAGAATGTGT-1,true,AAACCTGAGAATGTGT-1_contig_1,true,511,TRA,TRAV9-2,,TRAJ52,TRAC,true,false,CVLVTGANTGKLTF,TGTGTTTTGGTCACTGGAGCTAACACTGGAAAGCTCACGTTT,6096,4,clonotype100,clonotype100_consensus_1
AAACCTGAGAATGTGT-1,true,AAACCTGAGAATGTGT-1_contig_2,true,529,TRB,TRBV20,,TRBJ2-7,TRBC1,true,false,CGARGQNYEQYF,TGTGGTGCTCGGGGACAAAACTATGAACAGTACTTC,17762,10,clonotype100,clonotype100_consensus_2
AAACCTGAGGTGTGGT-1,true,AAACCTGAGGTGTGGT-1_contig_1,true,565,TRA,TRAV12D-3,,TRAJ12,TRAC,true,true,CALSGTGGYKVVF,TGTGCTCTGAGTGGGACTGGAGGCTATAAAGTGGTCTTT,13626,12,clonotype101,clonotype101_consensus_1
AAACCTGAGGTGTGGT-1,true,AAACCTGAGGTGTGGT-1_contig_2,true,489,TRB,TRBV3,TRBD1,TRBJ1-1,TRBC1,true,true,CASSLYGGPEVFF,TGTGCCAGCAGCTTATATGGGGGCCCAGAAGTCTTCTTT,16974,13,clonotype101,clonotype101_consensus_2"""

        file2_content = """barcode,is_cell,contig_id,high_confidence,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis,raw_clonotype_id,raw_consensus_id
AAACCTGAGGTGTGGT-1,true,AAACCTGAGGTGTGGT-1_contig_1,true,464,TRA,TRAV14N-1,,TRAJ5,TRAC,true,true,CAAKGTQVVGQLTF,TGTGCAGCAAAGGGGACACAGGTTGTGGGGCAGCTCACTTTC,16482,14,clonotype102,clonotype102_consensus_1
AAACCTGAGTACGTAA-1,true,AAACCTGAGTACGTAA-1_contig_1,true,487,TRA,TRAV13D-2,,TRAJ37,TRAC,true,true,CAIVGNTGKLIF,TGTGCTATAGTAGGCAATACCGGAAAACTCATCTTT,10862,7,clonotype103,clonotype103_consensus_1
AAACCTGAGTACGTAA-1,true,AAACCTGAGTACGTAA-1_contig_2,true,515,TRB,TRBV3,,TRBJ1-2,TRBC1,true,true,CASSFATNSDYTF,TGTGCCAGCAGCTTCGCAACAAACTCCGACTACACCTTC,10234,8,clonotype103,clonotype103_consensus_2
AAACCTGCAATGTAAG-1,true,AAACCTGCAATGTAAG-1_contig_1,true,544,TRA,TRAV3D-3,,TRAJ31,TRAC,true,true,CAVSANSNNRIFF,TGCGCAGTCAGTGCGAATAGCAATAACAGAATCTTCTTT,2900,2,clonotype104,clonotype104_consensus_1"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def test_import_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_10xGenomics_rep")

        self.create_dumy_dataset(path, add_metadata=True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "tenx_genomics")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["path"] = path
        params["metadata_file"] = path / "metadata.csv"
        params["separator"] = "," # new true separator is \t but tests still based on old ,


        dataset = TenxGenomicsImport(params, "tenx_dataset_repertoire").import_dataset()

        self.assertEqual(2, dataset.get_example_count())

        self.assertEqual(len(dataset.repertoires[0].sequences()), 2)
        self.assertEqual(len(dataset.repertoires[1].sequences()), 4)

        self.assertEqual(dataset.repertoires[0].sequences()[0].sequence_aa, "ALSGTGGYKVV")
        self.assertListEqual([Chain.ALPHA.value, Chain.BETA.value], dataset.repertoires[0].data.locus.tolist())
        self.assertListEqual([12, 13], dataset.repertoires[0].data.umi_count.tolist())

        shutil.rmtree(path)

    def test_import_sequence_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_10xGenomics_seq")

        self.create_dumy_dataset(path, add_metadata=False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "tenx_genomics")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path
        params["separator"] = "," # new true separator is \t but tests still based on old ,


        dataset = TenxGenomicsImport(params, "tenx_dataset_sequence").import_dataset()

        self.assertEqual(6, dataset.get_example_count())

        data = dataset.get_data(1)
        for receptorseq in data:
            self.assertTrue(
                receptorseq.sequence_aa in ["ALSGTGGYKVV", "ASSLYGGPEVF", "AAKGTQVVGQLT", "AIVGNTGKLI", "ASSFATNSDYT", "AVSANSNNRIF"])

        shutil.rmtree(path)

    def test_import_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_10xGenomics_receptor")

        self.create_dumy_dataset(path, add_metadata=False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "tenx_genomics")
        params["is_repertoire"] = False
        params["paired"] = True
        params["result_path"] = path
        params["path"] = path
        params["sequence_file_size"] = 1
        params["receptor_chains"] = "TRA_TRB"
        params["separator"] = "," # new true separator is \t but tests still based on old ,


        dataset = TenxGenomicsImport(params, "tenx_dataset_receptor").import_dataset()

        self.assertEqual(2, dataset.get_example_count())

        data = dataset.get_data(1)
        for receptor in data:
            self.assertTrue(receptor.alpha.sequence_aa in ["ALSGTGGYKVV", "AIVGNTGKLI"])
            self.assertTrue(receptor.beta.sequence_aa in ["ASSLYGGPEVF", "ASSFATNSDYT"])

        shutil.rmtree(path)
