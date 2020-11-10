import shutil
from unittest import TestCase

from source.IO.dataset_import.TenxGenomicsImport import TenxGenomicsImport
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestTenxGenomicsImport(TestCase):

    def create_dumy_dataset(self, path, add_metadata):
        file1_content = """clonotype_id,consensus_id,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis
clonotype100,clonotype100_consensus_1,843,TRA,TRAV9-2,None,TRAJ52,TRAC,True,False,CVLVTGANTGKLTF,TGTGTTTTGGTCACTGGAGCTAACACTGGAAAGCTCACGTTT,8566,4
clonotype100,clonotype100_consensus_2,685,TRB,TRBV20,None,TRBJ2-7,TRBC1,True,False,CGARGQNYEQYF,TGTGGTGCTCGGGGACAAAACTATGAACAGTACTTC,74572,29
clonotype101,clonotype101_consensus_1,620,TRA,TRAV12D-3,None,TRAJ12,TRAC,True,True,CALSGTGGYKVVF,TGTGCTCTGAGTGGGACTGGAGGCTATAAAGTGGTCTTT,3396,2
clonotype101,clonotype101_consensus_2,759,TRB,TRBV3,TRBD1,TRBJ1-1,TRBC1,True,True,CASSLYGGPEVFF,TGTGCCAGCAGCTTATATGGGGGCCCAGAAGTCTTCTTT,18133,4"""

        file2_content = """clonotype_id,consensus_id,length,chain,v_gene,d_gene,j_gene,c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis
clonotype102,clonotype102_consensus_1,675,TRA,TRAV14N-1,None,TRAJ5,TRAC,True,True,CAAKGTQVVGQLTF,TGTGCAGCAAAGGGGACACAGGTTGTGGGGCAGCTCACTTTC,23380,4
clonotype103,clonotype103_consensus_1,572,TRA,TRAV13D-2,None,TRAJ37,TRAC,True,True,CAIVGNTGKLIF,TGTGCTATAGTAGGCAATACCGGAAAACTCATCTTT,23899,13
clonotype103,clonotype103_consensus_2,753,TRB,TRBV3,None,TRBJ1-2,TRBC1,True,True,CASSFATNSDYTF,TGTGCCAGCAGCTTCGCAACAAACTCCGACTACACCTTC,52713,28
clonotype104,clonotype104_consensus_1,680,TRA,TRAV3D-3,None,TRAJ31,TRAC,True,True,CAVSANSNNRIFF,TGCGCAGTCAGTGCGAATAGCAATAACAGAATCTTCTTT,31289,6"""

        with open(path + "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path + "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            with open(path + "metadata.csv", "w") as file:
                file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")


    def test_import_repertoire_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/io_10xGenomics/"
        PathBuilder.build(path)
        self.create_dumy_dataset(path, add_metadata=True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/", "tenx_genomics")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["path"] = path
        params["metadata_file"] = path + "metadata.csv"

        dataset = TenxGenomicsImport.import_dataset(params, "tenx_dataset_repertoire")

        self.assertEqual(2, dataset.get_example_count())

        self.assertEqual(len(dataset.repertoires[0].sequences), 2)
        self.assertEqual(len(dataset.repertoires[1].sequences), 4)

        self.assertEqual(dataset.repertoires[0].sequences[0].amino_acid_sequence, "ALSGTGGYKVV")
        self.assertListEqual([Chain.ALPHA, Chain.BETA], list(dataset.repertoires[0].get_chains()))
        self.assertListEqual([2,4], list(dataset.repertoires[0].get_counts()))

        shutil.rmtree(path)



    def test_import_sequence_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/io_10xGenomics/"
        PathBuilder.build(path)
        self.create_dumy_dataset(path, add_metadata=False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/", "tenx_genomics")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path
        params["sequence_file_size"] = 1

        dataset = TenxGenomicsImport.import_dataset(params, "tenx_dataset_sequence")

        self.assertEqual(6, dataset.get_example_count())
        self.assertEqual(6, len(dataset.get_filenames()))

        data = dataset.get_data(1)
        for receptorseq in data:
            self.assertTrue(receptorseq.amino_acid_sequence in ["ALSGTGGYKVV", "ASSLYGGPEVF", "AAKGTQVVGQLT", "AIVGNTGKLI", "ASSFATNSDYT", "AVSANSNNRIF"])

        shutil.rmtree(path)


    def test_import_receptor_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/io_10xGenomics/"
        PathBuilder.build(path)
        self.create_dumy_dataset(path, add_metadata=False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/", "tenx_genomics")
        params["is_repertoire"] = False
        params["paired"] = True
        params["result_path"] = path
        params["path"] = path
        params["sequence_file_size"] = 1
        params["receptor_chains"] = "TRA_TRB"

        dataset = TenxGenomicsImport.import_dataset(params, "tenx_dataset_receptor")

        self.assertEqual(2, dataset.get_example_count())
        self.assertEqual(2, len(dataset.get_filenames()))

        data = dataset.get_data(1)
        for receptor in data:
            self.assertTrue(receptor.alpha.amino_acid_sequence in ["ALSGTGGYKVV", "AIVGNTGKLI"])
            self.assertTrue(receptor.beta.amino_acid_sequence in ["ASSLYGGPEVF", "ASSFATNSDYT"])

        shutil.rmtree(path)

