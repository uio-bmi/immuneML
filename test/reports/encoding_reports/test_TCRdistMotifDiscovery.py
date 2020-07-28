import os
import shutil
from unittest import TestCase

from source.IO.dataset_import.GenericReceptorImport import GenericReceptorImport
from source.caching.CacheType import CacheType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.data_reports.TCRdistMotifDiscovery import TCRdistMotifDiscovery
from source.util.PathBuilder import PathBuilder


class TestTCRdistMotifDiscovery(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dataset(self, path):
        data = """subject,epitope,count,v_a_gene,j_a_gene,cdr3_a_aa,cdr3_a_nucseq,v_b_gene,j_b_gene,cdr3_b_aa,cdr3_b_nucseq,clone_id
mouse_subject0050,PA,2,TRAV7-3*01,TRAJ33*01,CAVSLDSNYQLIW,tgtgcagtgagcctcgatagcaactatcagttgatctgg,TRBV13-1*01,TRBJ2-3*01,CASSDFDWGGDAETLYF,tgtgccagcagtgatttcgactggggaggggatgcagaaacgctgtatttt,mouse_tcr0072.clone
mouse_subject0050,PA,6,TRAV6D-6*01,TRAJ56*01,CALGDRATGGNNKLTF,tgtgctctgggtgacagggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSPDRGEVFF,tgtgctagcagtccggacaggggtgaagtcttcttt,mouse_tcr0096.clone
mouse_subject0050,PA,1,TRAV6D-6*01,TRAJ49*01,CALGSNTGYQNFYF,tgtgctctgggctcgaacacgggttaccagaacttctatttt,TRBV29*01,TRBJ1-5*01,CASTGGGAPLF,tgtgctagcacagggggaggggctccgcttttt,mouse_tcr0276.clone
mouse_subject0050,PA,2,TRAV6-4*01,TRAJ34*02,CALAPSNTNKVVF,tgtgctctggccccttccaataccaacaaagtcgtcttt,TRBV2*01,TRBJ2-7*01,CASSQDPGDYEQYF,tgtgccagcagccaagatcctggggactatgaacagtacttc,mouse_tcr0269.clone
mouse_subject0050,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggtcccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-4*01,CASSLGGENTLYF,tgtgctagcagtttagggggggaaaacaccttgtacttt,mouse_tcr0285.clone
mouse_subject0050,PA,1,TRAV6D-6*01,TRAJ23*01,CALGKNYNQGKLIF,tgtgctctggggaagaattataaccaggggaagcttatcttt,TRBV29*01,TRBJ2-7*01,CASDRAGEQYF,tgtgctagcgacagggccggggaacagtacttc,mouse_tcr0080.clone
mouse_subject0050,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTL,tgtatcctgagagtaggggctactggaggcaataataagctgactctt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagtttggacaggggtgaagtcttcttt,mouse_tcr0167.clone
mouse_subject0050,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggttccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLSGYEQYF,tgtgctagcagtttatccggatatgaacagtacttc,mouse_tcr0234.clone
mouse_subject0050,PA,1,TRAV12N-3*01,TRAJ33*01,CALSDRSNYQLIW,tgtgctctgagtgatcggagcaactatcagttgatctgg,TRBV29*01,TRBJ1-4*01,CASDGGERLFF,tgtgctagcgacgggggcgaaagattatttttc,mouse_tcr0253.clone
mouse_subject0050,PA,3,TRAV12N-3*01,TRAJ34*02,CALSASNTNKVVF,tgtgctctgagtgcttccaataccaacaaagtcgtcttt,TRBV19*01,TRBJ2-1*01,CASSMGAEQFF,tgtgccagcagtatgggtgctgagcagttcttc,mouse_tcr0102.clone
mouse_subject0050,PA,1,TRAV16D/DV11*03,TRAJ42*01,CAMIRSGGSNAKLTF,tgtgctatgatccgttctggaggaagcaatgcaaagctaaccttc,TRBV13-3*01,TRBJ2-1*01,CASSGLGGQAEQFF,tgtgccagcagcggactgggggggcaggctgagcagttcttc,mouse_tcr0376.clone
mouse_subject0050,PA,1,TRAV9D-4*04,TRAJ37*01,CAVTITGNTGKLIF,tgtgctgtgactataacaggcaataccggaaaactcatcttt,TRBV5*01,TRBJ2-2*01,CASSQEDSDTGQLYF,tgtgccagcagccaagaagacagtgacaccgggcagctctacttt,mouse_tcr0114.clone
mouse_subject0007,PA,1,TRAV5-4*01,TRAJ22*01,CAASRAGSWQLIF,tgtgctgcaagtagggctggcagctggcaactcatcttt,TRBV4*01,TRBJ1-4*01,CASREGGFSNERLFF,tgtgccagcagggaagggggtttttccaacgaaagattatttttc,mouse_tcr0416.clone
mouse_subject0007,PA,1,TRAV12N-3*01,TRAJ34*02,CALSETNTNKVVF,tgtgctctgagtgagaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSQGDEQYF,tgtgctagcagccagggggatgaacagtacttc,mouse_tcr0414.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ49*01,CALGASNTGYQNFYF,tgtgctctgggtgcttcgaacacgggttaccagaacttctatttt,TRBV29*01,TRBJ2-7*01,CASSPDRGEQYF,tgtgctagcagtccggacaggggagaacagtacttc,mouse_tcr0453.clone
mouse_subject0007,PA,3,TRAV4D-3*03,TRAJ33*01,CAAEAGSNYQLIW,tgtgctgctgaggcgggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASTQGAEVFF,tgtgctagcacccagggggcagaagtcttcttt,mouse_tcr0417.clone
mouse_subject0007,PA,1,TRAV12N-3*01,TRAJ34*02,CALSKTNTNKVVF,tgtgctctgagtaagaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSWGGEQYF,tgtgctagcagttgggggggcgaacagtacttc,mouse_tcr0423.clone
mouse_subject0007,PA,3,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagcctggacaggggagaagtcttcttt,mouse_tcr0411.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ33*01,CALGAGSNYQLIW,tgtgctctgggggccggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSSGQEVFF,tgtgctagcagttcgggacaggaagtcttcttt,mouse_tcr0449.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSGGGEQYF,tgtgctagcagtggggggggcgaacagtacttc,mouse_tcr0110.clone
"""
        filename = path + 'data.csv'

        with open(filename, "w") as file:
            file.writelines(data)

        return filename

    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path + "tcrdist_motif_discovery/")
        dataset_path = self._create_dataset(path)

        dataset = GenericReceptorImport.import_dataset({"path": dataset_path,
                                                        "result_path": path + "dataset/",
                                                        "separator": ",",
                                                        "columns_to_load": ["subject", "epitope", "count", "v_a_gene", "j_a_gene", "cdr3_a_aa",
                                                                            "v_b_gene", "j_b_gene", "cdr3_b_aa", "clone_id", "cdr3_a_nucseq",
                                                                                                                             "cdr3_b_nucseq"],
                                                        "column_mapping": {
                                                            "cdr3_a_aa": "alpha_amino_acid_sequence",
                                                            "cdr3_b_aa": "beta_amino_acid_sequence",
                                                            "cdr3_a_nucseq": "alpha_nucleotide_sequence",
                                                            "cdr3_b_nucseq": "beta_nucleotide_sequence",
                                                            "v_a_gene": "alpha_v_gene",
                                                            "v_b_gene": "beta_v_gene",
                                                            "j_a_gene": "alpha_j_gene",
                                                            "j_b_gene": "beta_j_gene",
                                                            "clone_id": "identifier"
                                                        },
                                                        "chains": "ALPHA_BETA",
                                                        "region_type": "CDR3",
                                                        "file_size": 50000,
                                                        "organism": "mouse"}, 'd1')

        dataset = TCRdistEncoder(8).encode(dataset, EncoderParams(f"{path}result/", LabelConfiguration([Label("epitope")])))

        report = TCRdistMotifDiscovery(dataset, path + "report/", "report name", 8, 20)
        report.generate_report()

        shutil.rmtree(path)
