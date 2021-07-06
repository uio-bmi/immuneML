import os
import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.SingleLineReceptorImport import SingleLineReceptorImport
from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestSingleLineReceptorImport(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_import_repertoire_dataset(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "io_generic_receptor/")
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
mouse_subject0050,PA,1,TRAV7-2*01,TRAJ18*01,CAGDRGSALGRLHF,tgtgcaggggatagaggttcagccttagggaggctgcatttt,TRBV5*01,TRBJ2-7*01,CASSQDFEQYF,tgtgccagcagccaagattttgaacagtacttc,mouse_tcr0034.clone
mouse_subject0050,PA,1,TRAV5-1*01,TRAJ27*01,CSASKDTNTGKLTF,tgctcagcaagtaaggacaccaatacaggcaaattaaccttt,TRBV19*01,TRBJ2-7*01,CASSIARWDSYEQYF,tgtgccagcagtatagctaggtgggactcctatgaacagtacttc,mouse_tcr0369.clone
mouse_subject0050,PA,5,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaagggtagcaataacagaatcttcttt,TRBV19*03,TRBJ1-1*01,CASSWGSEVFF,tgtgccagcagttgggggtcagaagtcttcttt,mouse_tcr0028.clone
mouse_subject0050,PA,1,TRAV9N-2*01,TRAJ57*01,CVLSALDQGGSAKLIF,tgtgttctgagcgcgttagatcaaggagggtctgcgaagctcatcttt,TRBV31*01,TRBJ2-5*01,CAWSLSFNQDTQYF,tgtgcctggagtctatcctttaaccaagacacccagtacttt,mouse_tcr0240.clone
mouse_subject0050,PA,5,TRAV6D-6*01,TRAJ33*01,CALGGGSNYQLIW,tgtgctctgggtggagggagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSLGGEVFF,tgtgctagcagtttagggggagaagtcttcttt,mouse_tcr0039.clone
mouse_subject0050,PA,1,TRAV9N-2*01,TRAJ45*01,CVLSARAEGADRLTF,tgtgttttgagcgcgagggcagaaggtgcagatagactcaccttt,TRBV2*01,TRBJ2-7*01,CASSQAGDSYEQYF,tgtgccagcagccaagcgggggactcctatgaacagtacttc,mouse_tcr0143.clone
mouse_subject0050,PA,1,TRAV3D-3*02,TRAJ26*01,CAVSVRNYAQGLTF,tgcgcagtcagtgttcggaactatgcccagggattaaccttc,TRBV13-3*01,TRBJ1-4*01,CASTDSNERLFF,tgtgccagcacagattccaacgaaagattatttttc,mouse_tcr0127.clone
mouse_subject0050,PA,1,TRAV7-4*01,TRAJ24*01,CAASGGTTASLGKLQF,tgtgcagctagtggtgggacaactgccagtttggggaaactgcagttt,TRBV1*01,TRBJ2-4*01,CTCSADENTLYF,tgcacctgcagtgcagacgaaaacaccttgtacttt,mouse_tcr0224.clone
mouse_subject0050,PA,3,TRAV13-1*01,TRAJ43*01,CALERDNNAPRF,tgtgctttggaacgggacaacaatgccccacgattt,TRBV29*01,TRBJ2-7*01,CASSLSGYEQYF,tgtgctagcagtttatccggatatgaacagtacttc,mouse_tcr0023.clone
mouse_subject0050,PA,2,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcagggggagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtatagggggagaagtcttcttt,mouse_tcr0065.clone
mouse_subject0050,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggtcccttccaataccaacaaagtcgtcttt,TRBV14*01,TRBJ2-4*01,CASSFWGALVQNTLYF,tgtgccagcagtttttggggggcgcttgttcaaaacaccttgtacttt,mouse_tcr0094.clone
mouse_subject0050,PA,2,TRAV9N-2*01,TRAJ21*01,CVLRRSNYNVLYF,tgtgttttgaggcggtctaattacaacgtgctttacttc,TRBV4*01,TRBJ2-4*01,CASYGTGQNTLYF,tgtgccagctatggaacaggtcaaaacaccttgtacttt,mouse_tcr0317.clone
mouse_subject0050,PA,4,TRAV12N-3*01,TRAJ34*02,CALSGSNTNKVVF,tgtgctctgagtggttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASAGGDEQYF,tgtgctagcgcgggaggggatgaacagtacttc,mouse_tcr0004.clone
mouse_subject0050,PA,1,TRAV21/DV12*01,TRAJ53*01,CILVGGSNYKLTF,tgtatcctggtcggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSSGREVFF,tgtgctagcagttccgggagggaagtcttcttt,mouse_tcr0329.clone
mouse_subject0050,PA,4,TRAV12N-3*01,TRAJ22*01,CALSASSGSWQLIF,tgtgctctgagtgcatcttctggcagctggcaactcatcttt,TRBV19*01,TRBJ2-7*01,CASSIARWDSYEQYF,tgtgccagcagtatagctaggtgggactcctatgaacagtacttc,mouse_tcr0125.clone
mouse_subject0050,PA,3,TRAV12N-3*01,TRAJ12*01,CALSRTGGYKVVF,tgtgctctgagtcggactggaggctataaagtggtcttt,TRBV29*01,TRBJ2-7*01,CASSWGDEQYF,tgtgctagcagttggggggatgaacagtacttc,mouse_tcr0008.clone
mouse_subject0050,PA,12,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagtttggacaggggtgaagtcttcttt,mouse_tcr0020.clone
mouse_subject0050,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgaagggagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcatatgaacagtacttc,mouse_tcr0256.clone
mouse_subject0050,PA,9,TRAV6D-6*01,TRAJ32*01,CALGAGSGNKLIF,tgtgctctgggtgccggcagtggcaacaagctcatcttt,TRBV29*01,TRBJ1-1*01,CASSWDRGEVFF,tgtgctagcagttgggacaggggggaagtcttcttt,mouse_tcr0169.clone
mouse_subject0050,PA,1,TRAV9N-2*01,TRAJ57*01,CVLSALDQGGSAKLIF,tgtgttctgagcgcgttggatcaaggagggtctgcgaagctcatcttt,TRBV31*01,TRBJ2-5*01,CAWSLSFNQDTQYF,tgtgcctggagtctatcctttaaccaagacacccagtacttt,mouse_tcr0266.clone
mouse_subject0050,PA,1,TRAV6D-6*01,TRAJ53*01,CALVGGSNYKLTF,tgtgctctggttggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSPTGEVFF,tgtgctagcagtccgacaggggaagtcttcttt,mouse_tcr0116.clone
mouse_subject0050,PA,2,TRAV6D-6*03,TRAJ53*01,CALIGGSNYKLTF,tgcgctctgattggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-2*01,CASTDTGQLYF,tgtgctagcaccgacaccgggcagctctacttt,mouse_tcr0178.clone
mouse_subject0050,PA,7,TRAV12N-3*01,TRAJ30*01,CALSRTNAYKVIF,tgtgctctgagtcgcacaaatgcttacaaagtcatcttt,TRBV29*01,TRBJ2-7*01,CASSLTDEQYF,tgtgctagcagtttaacggatgaacagtacttc,mouse_tcr0016.clone
mouse_subject0006,PA,4,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRAEVFF,tgtgctagcagtttagacagggcggaagtcttcttt,mouse_tcr0485.clone
mouse_subject0006,PA,7,TRAV6D-6*01,TRAJ40*01,CALGDRRTGNYKYVF,tgtgctctgggtgatcgtcggacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-7*01,CASSLDRGEQYF,tgtgctagcagtttagacaggggtgaacagtacttc,mouse_tcr0488.clone
mouse_subject0006,PA,5,TRAV12N-3*01,TRAJ12*01,CALSRTGGYKVVF,tgtgctctgagtaggactggaggctataaagtggtcttt,TRBV29*01,TRBJ2-7*01,CASSWGDEQYF,tgtgctagcagctggggggatgaacagtacttc,mouse_tcr0490.clone
mouse_subject0006,PA,2,TRAV12N-3*01,TRAJ34*02,CALSRSNTNKVVF,tgtgctctgagtaggtccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSTGDEQYF,tgtgctagcagtacaggggatgaacagtacttc,mouse_tcr0489.clone
mouse_subject0006,PA,2,TRAV3-1*01,TRAJ40*01,CAVSAGGNYKYVF,tgcgcagtcagtgcaggaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-5*02,CASTQGGAPLF,tgtgctagcacccagggcggagctccgcttttt,mouse_tcr0502.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSGGGEQYF,tgtgctagcagcggggggggggaacagtacttc,mouse_tcr0300.clone
mouse_subject0051,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*03,TRBJ2-7*01,CASAEGGEQYF,tgtgccagcgccgaggggggggaacagtacttc,mouse_tcr0036.clone
mouse_subject0051,PA,1,TRAV9D-4*04,TRAJ33*01,CAVAPGSNYQLIW,tgtgctgtggcccctggaagcaactatcagttgatctgg,TRBV29*01,TRBJ2-4*01,CASSWGDTLYF,tgtgctagcagctggggggacaccttgtacttt,mouse_tcr0279.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ56*01,CALGGGATGGNNKLTF,tgtgctctgggggggggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASTPDRGEVFF,tgtgctagcaccccggacaggggagaagtcttcttt,mouse_tcr0153.clone
mouse_subject0051,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggtcccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ1-1*01,CASSQAEVFF,tgtgctagcagccaggcagaagtcttcttt,mouse_tcr0121.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctggggggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLGDEQYF,tgtgctagcagtttaggggatgaacagtacttc,mouse_tcr0128.clone
mouse_subject0051,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcagggggagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtataggcggagaagtcttcttt,mouse_tcr0305.clone
mouse_subject0051,PA,1,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaagggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGGEQYF,tgtgccagcagtatagggggggaacagtacttc,mouse_tcr0341.clone
mouse_subject0051,PA,1,TRAV10*01,TRAJ22*01,CAASSGSWQLIF,tgtgcagcatcttctggcagctggcaactcatcttt,TRBV5*01,TRBJ2-4*01,CASSPTGGDQNTLYF,tgtgccagcagcccgacagggggcgatcaaaacaccttgtacttt,mouse_tcr0311.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-4*01,CASSQGERLFF,tgtgctagcagccagggagagagattatttttc,mouse_tcr0211.clone
mouse_subject0051,PA,1,TRAV12D-2*02,TRAJ34*02,CALTGPSNTNKVVF,tgtgctttgaccggtccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ1-1*01,CASSWATEVFF,tgtgctagcagttgggcaacagaagtcttcttt,mouse_tcr0146.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ9*02,CALGRGMGYKLTF,tgtgctctgggtcggggcatgggctacaaacttaccttc,TRBV29*01,TRBJ1-4*01,CASSLGDGLFF,tgtgctagcagtttaggggacggattatttttc,mouse_tcr0236.clone
mouse_subject0051,PA,3,TRAV9N-2*01,TRAJ15*01,CVLSAWGGRALIF,tgtgttctgagcgcgtggggaggcagagctctgatattt,TRBV17*01,TRBJ2-2*01,CASSREEANTGQLYF,tgtgctagcagtagagaggaggccaacaccgggcagctctacttt,mouse_tcr0204.clone
mouse_subject0051,PA,1,TRAV6D-6*03,TRAJ22*01,CALIQSSGSWQLIF,tgcgctctgatccaatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ1-1*01,CASSFGGEVFF,tgtgctagcagtttcgggggtgaagtcttcttt,mouse_tcr0054.clone
mouse_subject0051,PA,4,TRAV17*02,TRAJ39*01,CALDGNNAGAKLTF,tgtgcactggacggcaataatgcaggtgccaagctcacattc,TRBV13-2*01,TRBJ2-5*01,CASGDPDRGNQDTQYF,tgtgccagcggtgatccggacaggggtaaccaagacacccagtacttt,mouse_tcr0047.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ22*01,CALARSSGSWQLIF,tgtgctctggctcgatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ2-1*01,CASSPDWGAF,tgtgctagcagcccggactggggggccttc,mouse_tcr0278.clone
mouse_subject0051,PA,1,TRAV9-4*01,TRAJ22*01,CAFFASSGSWQLIF,tgtgctttcttcgcatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ1-4*01,CASSFGKRLFF,tgtgctagcagttttgggaaaagattatttttc,mouse_tcr0150.clone
mouse_subject0051,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagtttggacaggggagaagtcttcttt,mouse_tcr0149.clone
mouse_subject0051,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*03,TRBJ2-7*01,CASSFGDEQYF,tgtgccagcagttttggggatgaacagtacttc,mouse_tcr0033.clone
mouse_subject0051,PA,2,TRAV10*01,TRAJ7*01,CAASTAYSNNRLTL,tgtgcagcaagcacggcctacagcaacaacagacttactttg,TRBV13-3*01,TRBJ2-1*01,CASSDGGGYNYAEQFF,tgtgccagcagtgacggcggggggtataactatgctgagcagttcttc,mouse_tcr0247.clone
mouse_subject0051,PA,1,TRAV9-1*01,TRAJ12*01,CAVSAPGGYKVVF,tgtgctgtgagcgcgcctggaggctataaagtggtcttt,TRBV20*01,TRBJ1-3*01,CGDGTGGNTLYF,tgtggtgacgggacagggggaaatacgctctatttt,mouse_tcr0274.clone
mouse_subject0051,PA,1,TRAV12N-3*01,TRAJ40*01,CALSNTGNYKYVF,tgtgctctgagtaatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-7*01,CASSFGGQQYF,tgtgctagcagttttgggggacaacagtacttc,mouse_tcr0228.clone
mouse_subject0051,PA,1,TRAV12D-3*02,TRAJ40*01,CALSYTGNYKYVF,tgtgctttgagttatacaggaaactacaaatacgtcttt,TRBV2*01,TRBJ1-2*01,CASSQEGGGDSDYTF,tgtgccagcagccaagagggggggggcgactccgactacaccttc,mouse_tcr0306.clone
mouse_subject0051,PA,1,TRAV9N-2*01,TRAJ21*01,CVLSAPMSNYNVLYF,tgtgttctgagcgcccctatgtctaattacaacgtgctttacttc,TRBV19*01,TRBJ2-5*01,CASSIAGAQDTQYF,tgtgccagcagtatagctggggcccaagacacccagtacttt,mouse_tcr0173.clone
mouse_subject0051,PA,1,TRAV16D/DV11*03,TRAJ53*01,CAMREGSGGSNHKLTF,tgtgctatgagagagggcagtggaggcagcaatcacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLTPEEVFF,tgtgctagcagtttaactccagaagaagtcttcttt,mouse_tcr0250.clone
mouse_subject0051,PA,1,TRAV12D-2*02,TRAJ16*01,CALSDRPSSGQKLVF,tgtgctttgagtgatcggccttcaagtggccagaagctggttttt,TRBV2*01,TRBJ1-2*01,CASSQEGAGDSDYTF,tgtgccagcagccaagagggggcaggggactccgactacaccttc,mouse_tcr0264.clone
mouse_subject0051,PA,1,TRAV9N-2*01,TRAJ30*01,CVLSARTNAYKVIF,tgtgttttgagcgcgagaacaaatgcttacaaagtcatcttt,TRBV2*01,TRBJ1-4*01,CASSQDKDFSNERLFF,tgtgccagcagccaagataaagatttttccaacgaaagattatttttc,mouse_tcr0168.clone
mouse_subject0051,PA,3,TRAV12N-3*01,TRAJ6*01,CALSKTSGGNYKPTF,tgtgctctgagtaagacctcaggaggaaactacaaacctacgttt,TRBV29*01,TRBJ2-1*01,CASSPDGEQFF,tgtgctagcagtcctgacggtgagcagttcttc,mouse_tcr0005.clone
mouse_subject0051,PA,5,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgagggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSSGEGVFF,tgtgctagcagttcaggggagggagtcttcttt,mouse_tcr0042.clone
mouse_subject0051,PA,1,TRAV9N-2*01,TRAJ15*01,CVLSAWGGRALIF,tgtgttttgagcgcgtggggaggcagagctctgatattt,TRBV17*01,TRBJ2-5*01,CASSRETGNQDTQYF,tgtgctagcagtagagaaacaggtaaccaagacacccagtacttt,mouse_tcr0158.clone
mouse_subject0051,PA,7,TRAV7-6*01,TRAJ33*01,CAVSPGSNYQLIW,tgtgcagtgagccccggtagcaactatcagttgatctgg,TRBV20*01,TRBJ1-3*01,CGDGTGGNTLYF,tgtggtgacgggacagggggaaatacgctctatttt,mouse_tcr0026.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ56*01,CALGDGATGGNNKLTF,tgtgctctgggtgatggggctactggaggcaataataagctgactttt,TRBV17*01,TRBJ2-5*01,CASGREDTQYF,tgtgctagcggcagggaagacacccagtacttt,mouse_tcr0345.clone
mouse_subject0051,PA,5,TRAV6D-6*01,TRAJ33*01,CALGTGSNYQLIW,tgtgctctgggtaccggtagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSSGTGDF,tgtgctagcagttccgggacaggggacttc,mouse_tcr0019.clone
mouse_subject0051,PA,1,TRAV13-1*01,TRAJ53*01,CALYSGGSNYKLTF,tgtgctttgtacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CANSQGTEVFF,tgtgctaacagtcagggaacagaagtcttcttt,mouse_tcr0221.clone
mouse_subject0051,PA,7,TRAV12N-3*01,TRAJ34*02,CALSASNTNKVVF,tgtgctctgagtgcttccaataccaacaaagtcgtcttt,TRBV19*01,TRBJ2-1*01,CASSMGAEQFF,tgtgccagcagtatgggcgctgagcagttcttc,mouse_tcr0097.clone
mouse_subject0051,PA,5,TRAV6-4*01,TRAJ34*02,CALAPSNTNKVVF,tgtgctctggcgccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSYGGGEQYF,tgtgctagcagttatgggggcggggaacagtacttc,mouse_tcr0017.clone
mouse_subject0051,PA,1,TRAV13-1*01,TRAJ53*01,CALYSGGSNYKLTF,tgtgctttgtacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSQGTEVFF,tgtgctagcagtcagggaacagaagtcttcttt,mouse_tcr0144.clone
mouse_subject0051,PA,1,TRAV6D-6*01,TRAJ53*01,CALGEGSNYKLTF,tgtgctctgggtgaaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLGGEQYF,tgtgctagcagtttagggggcgaacagtacttc,mouse_tcr0262.clone
mouse_subject0051,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGNEQYF,tgtgccagcagtattgggaatgaacagtacttc,mouse_tcr0301.clone
mouse_subject0051,PA,3,TRAV9N-3*01,TRAJ21*01,CAVRMSNYNVLYF,tgtgctgtgaggatgtctaattacaacgtgctttacttc,TRBV4*01,TRBJ2-1*01,CASSYPDIYAEQFF,tgtgccagcagctacccggacatctatgctgagcagttcttc,mouse_tcr0002.clone
mouse_subject0051,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctcagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSSAPEQYF,tgtgctagcagttcggcccctgaacagtacttc,mouse_tcr0134.clone
mouse_subject0051,PA,4,TRAV12D-3*02,TRAJ40*01,CALSPPNTGNYKYVF,tgtgctttgagccctcctaatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-1*01,CASSLDRGWVFF,tgtgctagcagtttggacaggggatgggtcttcttt,mouse_tcr0009.clone
mouse_subject0051,PA,2,TRAV6D-6*03,TRAJ42*01,CALSGGSNAKLTF,tgcgctctgagtggaggaagcaatgcaaagctaaccttc,TRBV29*01,TRBJ1-1*01,CASSDGLPFF,tgtgctagcagtgacgggctccccttcttt,mouse_tcr0063.clone
mouse_subject0051,PA,1,TRAV16D/DV11*03,TRAJ53*01,CAMREGSGGSNYKLTF,tgtgctatgagagagggcagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLTPEEVFF,tgtgctagcagtttaactccagaagaagtcttcttt,mouse_tcr0315.clone
mouse_subject0051,PA,2,TRAV11*02,TRAJ33*01,CVVGSNYQLIW,tgtgtggtgggaagcaactatcagttgatctgg,TRBV19*01,TRBJ2-5*01,CASSIGLGLEDTQYF,tgtgccagcagtataggactgggactagaagacacccagtacttt,mouse_tcr0035.clone
mouse_subject0051,PA,5,TRAV12N-3*01,TRAJ34*02,CALSGSNTNKVVF,tgtgctctgagtggttccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-7*01,CASSSGGEQYF,tgtgccagcagttcagggggggaacagtacttc,mouse_tcr0001.clone
mouse_subject0021,PA,2,TRAV6D-6*01,TRAJ6*01,CALVSGGNYKPTF,tgtgctctggtgtcaggaggaaactacaaacctacgttt,TRBV29*01,TRBJ1-5*01,CASSWGGAPLF,tgtgctagcagttgggggggggctccgcttttt,mouse_tcr2082.clone
mouse_subject0021,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGDEQYF,tgtgccagcagtataggggatgaacagtacttc,mouse_tcr2081.clone
mouse_subject0021,PA,1,TRAV6D-6*01,TRAJ33*01,CALGDGRNYQLIW,tgtgctctgggtgatggtcgcaactatcagttgatctgg,TRBV29*01,TRBJ1-3*01,CASTSGNTLYF,tgtgctagcacttctggaaatacgctctatttt,mouse_tcr2093.clone
mouse_subject0021,PA,1,TRAV7-4*01,TRAJ40*01,CAASEGGNYKYVF,tgtgcagctagtgaaggaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-7*01,CASSGGNEQYF,tgtgctagcagtggagggaatgaacagtacttc,mouse_tcr2080.clone
mouse_subject0021,PA,1,TRAV9N-2*01,TRAJ53*01,CVCYSGGSNYKLTF,tgtgtttgttacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLGGGVFF,tgtgctagcagtttaggtgggggagtcttcttt,mouse_tcr2088.clone
mouse_subject0021,PA,1,TRAV9N-2*01,TRAJ53*01,CVLSGGSNYKLTF,tgtgttttgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSFGGAPLF,tgtgctagcagtttcgggggggctccgcttttt,mouse_tcr2098.clone
mouse_subject0021,PA,1,TRAV6D-6*01,TRAJ33*01,CALGAGSNYQLIW,tgtgctctgggtgcgggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSLGRGVFF,tgtgctagcagtttagggcggggagtcttcttt,mouse_tcr2090.clone
mouse_subject0021,PA,2,TRAV6D-6*01,TRAJ53*01,CALVGGSNYKLTF,tgtgctctggttggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSFGTEVFF,tgtgctagcagtttcggcacagaagtcttcttt,mouse_tcr2083.clone
mouse_subject0021,PA,1,TRAV5-4*01,TRAJ43*01,CAAVPNNNNAPRF,tgtgctgccgtccccaataacaacaatgccccacgattt,TRBV29*01,TRBJ2-5*01,CASEGGPNQDTQYF,tgtgctagcgaagggggtcctaaccaagacacccagtacttt,mouse_tcr2089.clone
mouse_subject0021,PA,2,TRAV6-4*01,TRAJ21*01,CALAPSNYNVLYF,tgtgctctggccccgtctaattacaacgtgctttacttc,TRBV29*01,TRBJ2-7*01,CASSLSGFEQYF,tgtgctagcagtttatcgggatttgaacagtacttc,mouse_tcr2092.clone
mouse_subject0021,PA,1,TRAV1*01,TRAJ23*01,CAVDYNQGKLIF,tgtgctgtcgattataaccaggggaagcttatcttt,TRBV14*01,TRBJ2-5*01,CASSPLGGRRDTQYF,tgtgccagcagtcctctgggggggcgcagggacacccagtacttt,mouse_tcr2095.clone
mouse_subject0021,PA,1,TRAV12N-3*01,TRAJ40*01,CALSHTGNYKYVF,tgtgctctgagtcatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-7*01,CASSQGGRQYF,tgtgctagcagtcagggaggacgacagtacttc,mouse_tcr2086.clone
mouse_subject0021,PA,1,TRAV6D-6*01,TRAJ33*01,CALGLGSNYQLIW,tgtgctctggggctggggagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSLGGEVFF,tgtgctagcagtttggggggagaagtcttcttt,mouse_tcr2079.clone
mouse_subject0021,PA,4,TRAV9N-2*01,TRAJ45*01,CVLSARTEGADRLTF,tgtgttttgagcgcgaggacagaaggtgcagatagactcaccttt,TRBV2*01,TRBJ2-7*01,CASSQQQDSYEQYF,tgtgccagcagccaacaacaggactcctatgaacagtacttc,mouse_tcr2084.clone
mouse_subject0049,PA,2,TRAV12N-3*01,TRAJ40*01,CALSYTGNYKYVF,tgtgctctgagttatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-5*01,CASSLGQAPLF,tgtgctagcagtttaggccaggctccgcttttt,mouse_tcr0022.clone
mouse_subject0049,PA,1,TRAV7-6*02,TRAJ33*01,CAVSMDSNYQLIW,tgtgcagtgagcatggatagcaactatcagttgatctgg,TRBV13-1*01,TRBJ2-2*01,CASSEWDWGGEGTGQLYF,tgtgccagcagtgaatgggactgggggggcgagggcaccgggcagctctacttt,mouse_tcr0319.clone
mouse_subject0049,PA,1,TRAV9-4*01,TRAJ34*02,CALSPHTNKVVF,tgtgctctgagcccccataccaacaaagtcgtcttt,TRBV17*01,TRBJ2-1*01,CASSRGWGGPYNYAEQFF,tgtgctagcagtagaggctgggggggcccctataactatgctgagcagttcttc,mouse_tcr0032.clone
mouse_subject0049,PA,1,TRAV12D-3*02,TRAJ30*01,CALSHTNAYKVIF,tgtgctttgtctcacacaaatgcttacaaagtcatcttt,TRBV29*01,TRBJ2-7*01,CASSWGGEQYF,tgtgctagcagttgggggggcgaacagtacttc,mouse_tcr0195.clone
mouse_subject0049,PA,1,TRAV9N-2*01,TRAJ33*01,CVLIMDSNYQLIW,tgtgttttgatcatggatagcaactatcagttgatctgg,TRBV29*01,TRBJ1-5*01,CASSLGQAPLF,tgtgctagcagtttagggcaggctccgcttttt,mouse_tcr0192.clone
mouse_subject0049,PA,1,TRAV7-4*01,TRAJ24*01,CAASGGTTASLGKLQF,tgtgcagctagtggggggacaactgccagtttggggaaactgcagttt,TRBV1*01,TRBJ2-4*01,CTCSAEGNTLYF,tgcacctgcagtgcagagggaaacaccttgtacttt,mouse_tcr0111.clone
mouse_subject0049,PA,3,TRAV12D-3*02,TRAJ34*02,CALSGVNTNKVVF,tgtgctttgagtggagtcaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-1*01,CASNQGGEQFF,tgtgccagcaatcaggggggtgagcagttcttc,mouse_tcr0058.clone
mouse_subject0049,PA,1,TRAV6D-6*03,TRAJ6*01,CALIPSGGNYKPTF,tgcgctctgatcccctcaggaggaaactacaaacctacgttt,TRBV29*01,TRBJ2-5*01,CASWSGPQDTQYF,tgtgctagctggtcagggccccaagacacccagtacttt,mouse_tcr0223.clone
mouse_subject0049,PA,1,TRAV9N-2*01,TRAJ23*01,CVLSAGYNQGKLIF,tgtgttttgagcgcagggtataaccaggggaagcttatcttt,TRBV13-3*01,TRBJ2-7*01,CASSDRHYEQYF,tgtgccagcagtgatcggcattatgaacagtacttc,mouse_tcr0071.clone
mouse_subject0049,PA,1,TRAV13-4/DV7*02,TRAJ43*01,CAMEQTNNNNAPRF,tgtgctatggaacaaaccaataacaacaatgccccacgattt,TRBV29*01,TRBJ1-5*01,CASSLGQAPLF,tgtgcaagcagtttaggccaggctccgcttttt,mouse_tcr0304.clone
mouse_subject0049,PA,1,TRAV10*01,TRAJ2*01,CAASRNTGGLSGKLTF,tgtgcagcaagccgtaatactggaggactaagtggtaaattaacattc,TRBV17*01,TRBJ2-2*01,CASSNQDRDYTGQLYF,tgtgctagcagtaaccaggacagggattacaccgggcagctctacttt,mouse_tcr0297.clone
mouse_subject0049,PA,1,TRAV9-4*01,TRAJ15*01,CALAYQGGRALIF,tgtgctctggcctaccagggaggcagagctctgatattt,TRBV17*01,TRBJ2-1*01,CASSRERDRDAEQFF,tgtgctagcagtagagaacgggacagggatgctgagcagttcttc,mouse_tcr0355.clone
mouse_subject0049,PA,2,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagtagttatgaacagtacttc,mouse_tcr0147.clone
mouse_subject0049,PA,4,TRAV12N-3*01,TRAJ34*02,CALSASNTNKVVF,tgtgctctgagtgcttccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-1*01,CASSGGAEQFF,tgtgccagcagtgggggtgctgagcagttcttc,mouse_tcr0012.clone
mouse_subject0049,PA,2,TRAV6D-6*01,TRAJ53*01,CALCGGSNYKLTF,tgtgctctttgtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagctcctatgaacagtacttc,mouse_tcr0122.clone
mouse_subject0049,PA,2,TRAV9N-2*01,TRAJ15*01,CVLAYQGGRALIF,tgtgttctggcctaccagggaggcagagctctgatattt,TRBV17*01,TRBJ2-1*01,CASSRETGGAEQFF,tgtgctagcagtagagagacagggggcgctgagcagttcttc,mouse_tcr0155.clone
mouse_subject0049,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSSGGEVFF,tgtgctagcagttcggggggagaagtcttcttt,mouse_tcr0175.clone
mouse_subject0049,PA,2,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaagggtagcaataacagaatcttcttt,TRBV19*03,TRBJ1-1*01,CASASGGEVFF,tgtgccagcgcatcagggggagaagtcttcttt,mouse_tcr0154.clone
mouse_subject0049,PA,1,TRAV12N-3*01,TRAJ34*02,CALSASNTNKVVF,tgtgctctgagtgcctccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ1-1*01,CASSWGGEVFF,tgtgccagcagttgggggggagaagtcttcttt,mouse_tcr0340.clone
mouse_subject0049,PA,3,TRAV4D-4*03,TRAJ43*01,CAAEAGNNNNAPRF,tgtgctgctgaggccggcaataacaacaatgccccacgattt,TRBV19*03,TRBJ1-1*01,CASSPDITEVFF,tgtgccagcagtcctgacatcacagaagtcttcttt,mouse_tcr0038.clone
mouse_subject0049,PA,1,TRAV6D-6*01,TRAJ56*01,CALGDRATGGNNKLTF,tgtgctctgggtgatagagctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSPDRGEVFF,tgtgctagcagtccggacaggggagaagtcttcttt,mouse_tcr0281.clone
mouse_subject0049,PA,5,TRAV4D-3*03,TRAJ33*01,CAAEAGSNYQLIW,tgtgctgctgaggcaggcagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSFGGEVFF,tgtgctagcagtttcgggggagaagtcttcttt,mouse_tcr0135.clone
mouse_subject0049,PA,1,TRAV6D-6*01,TRAJ33*01,CALGGGSNYQLIW,tgtgctctgggtggaggtagcaactatcagttgatctgg,TRBV29*01,TRBJ2-3*01,CASSGGETLYF,tgtgctagctcaggaggggaaacgctgtatttt,mouse_tcr0176.clone
mouse_subject0049,PA,1,TRAV12N-3*01,TRAJ34*02,CALSRSNTNKVVF,tgtgctctgagccgttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-1*01,CASSSGGGQFF,tgtgctagcagttcagggggggggcagttcttc,mouse_tcr0202.clone
mouse_subject0049,PA,11,TRAV16D/DV11*03,TRAJ42*01,CAMIRSGGSNAKLTF,tgtgctatgatccgttctggaggaagcaatgcaaagctaaccttc,TRBV31*01,TRBJ1-1*01,CAWSPGREVFF,tgtgcctggagtccagggagagaagtcttcttt,mouse_tcr0043.clone
mouse_subject0049,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*03,TRBJ2-7*01,CASSLGNEQYF,tgtgccagcagtctagggaatgaacagtacttc,mouse_tcr0183.clone
mouse_subject0049,PA,8,TRAV21/DV12*01,TRAJ44*01,CILTSGVTGSGGKLTL,tgtatcctgacctcaggggttactggcagtggtggaaaactcactttg,TRBV29*01,TRBJ1-1*01,CASSWDRGEVFF,tgtgctagcagttgggacaggggagaagtcttcttt,mouse_tcr0006.clone
mouse_subject0049,PA,3,TRAV18*01,TRAJ58*01,CTASMQQGTGSKLSF,tgtacagcctcgatgcagcaaggcactgggtctaagctgtcattt,TRBV29*01,TRBJ1-1*01,CASSWGTEVFF,tgtgctagcagttggggaacagaagtcttcttt,mouse_tcr0214.clone
mouse_subject0007,PA,1,TRAV4D-3*03,TRAJ27*01,CAAEEGTNTGKLTF,tgtgctgctgaggagggcaccaatacaggcaaattaaccttt,TRBV29*01,TRBJ2-3*01,CASSEDREFSAETLYF,tgtgctagcagtgaagacagggagtttagtgcagaaacgctgtatttt,mouse_tcr0412.clone
mouse_subject0007,PA,2,TRAV6D-6*01,TRAJ53*01,CALCGGSNYKLTF,tgtgctctgtgtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSEYEQYF,tgtgctagcagtgaatatgaacagtacttc,mouse_tcr0410.clone
mouse_subject0007,PA,10,TRAV9-1*01,TRAJ12*01,CAVSASGGYKVVF,tgtgctgtgagcgcgtctggaggctataaagtggtcttt,TRBV13-3*01,TRBJ2-7*01,CASSELGGQGEQYF,tgtgccagcagtgaactgggggggcagggagaacagtacttc,mouse_tcr0426.clone
mouse_subject0007,PA,1,TRAV9N-2*01,TRAJ49*01,CVFGGYQNFYF,tgtgtttttgggggttaccagaacttctatttt,TRBV24*02,TRBJ2-4*01,CASSQDRNTLYF,tgtgccagcagtcaggacagaaacaccttgtacttt,mouse_tcr0448.clone
mouse_subject0007,PA,2,TRAV6-4*01,TRAJ34*02,CALAPSNTNKVVF,tgtgctctggccccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLSGFEQYF,tgtgctagcagtttatcagggtttgaacagtacttc,mouse_tcr0454.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ38*01,CALALGDNSKLIW,tgtgctctggcccttggtgacaacagtaagctgatttgg,TRBV29*01,TRBJ2-7*01,CASSSGDF,tgtgctagcagttcgggggacttc,mouse_tcr0437.clone
mouse_subject0007,PA,2,TRAV15D-1/DV6D-1*04,TRAJ30*01,CALWEPDTNAYKVIF,tgtgctctctgggagcctgacacaaatgcttacaaagtcatcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtattgggggggaagtcttcttt,mouse_tcr0413.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ56*01,CALGSGGNNKLTF,tgtgctctgggttctggaggcaataataagctgactttt,TRBV29*01,TRBJ2-7*01,CASSPYEQYF,tgtgctagcagcccctatgaacagtacttc,mouse_tcr0415.clone
mouse_subject0007,PA,8,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggtcccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLSGFEQYF,tgtgctagcagtttatcagggtttgaacagtacttc,mouse_tcr0427.clone
mouse_subject0007,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctcagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSWGGEQYF,tgtgctagcagttgggggggcgaacagtacttc,mouse_tcr0425.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ33*01,CALGGGSNYQLIW,tgtgctctgggtggaggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSWGEEVFF,tgtgctagcagttggggagaggaagtcttcttt,mouse_tcr0431.clone
mouse_subject0007,PA,1,TRAV5-4*01,TRAJ22*01,CAASRAGSWQLIF,tgtgctgcaagtagggctggcagctggcaactcatcttt,TRBV4*01,TRBJ1-4*01,CASREGGFSNERLFF,tgtgccagcagggaagggggtttttccaacgaaagattatttttc,mouse_tcr0416.clone
mouse_subject0007,PA,1,TRAV12N-3*01,TRAJ34*02,CALSETNTNKVVF,tgtgctctgagtgagaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSQGDEQYF,tgtgctagcagccagggggatgaacagtacttc,mouse_tcr0414.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ49*01,CALGASNTGYQNFYF,tgtgctctgggtgcttcgaacacgggttaccagaacttctatttt,TRBV29*01,TRBJ2-7*01,CASSPDRGEQYF,tgtgctagcagtccggacaggggagaacagtacttc,mouse_tcr0453.clone
mouse_subject0007,PA,3,TRAV4D-3*03,TRAJ33*01,CAAEAGSNYQLIW,tgtgctgctgaggcgggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASTQGAEVFF,tgtgctagcacccagggggcagaagtcttcttt,mouse_tcr0417.clone
mouse_subject0007,PA,1,TRAV12N-3*01,TRAJ34*02,CALSKTNTNKVVF,tgtgctctgagtaagaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSWGGEQYF,tgtgctagcagttgggggggcgaacagtacttc,mouse_tcr0423.clone
mouse_subject0007,PA,3,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagcctggacaggggagaagtcttcttt,mouse_tcr0411.clone
mouse_subject0007,PA,1,TRAV6D-6*01,TRAJ33*01,CALGAGSNYQLIW,tgtgctctgggggccggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSSGQEVFF,tgtgctagcagttcgggacaggaagtcttcttt,mouse_tcr0449.clone
mouse_subject0007,PA,1,TRAV6-5*01,TRAJ27*01,CALSEPNTGKLTF,tgtgctctgagtgaacccaatacaggcaaattaaccttt,TRBV29*01,TRBJ1-1*01,CASSLGGPTEVFF,tgtgctagcagtttaggggggcccacagaagtcttcttt,mouse_tcr0439.clone
mouse_subject0052,PA,3,TRAV6D-6*01,TRAJ40*01,CALGDRGTGNYKYVF,tgtgctctgggtgatcggggtacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagtttagacaggggtgaagtcttcttt,mouse_tcr0164.clone
mouse_subject0052,PA,2,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSSGGEVFF,tgtgctagcagttcagggggcgaagtcttcttt,mouse_tcr0142.clone
mouse_subject0052,PA,1,TRAV6D-6*01,TRAJ6*01,CALGSGGNYKPTF,tgtgctctgggttcaggaggaaactacaaacctacgttt,TRBV29*01,TRBJ2-1*01,CASSWGAEQFF,tgtgctagcagttggggggctgagcagttcttc,mouse_tcr0074.clone
mouse_subject0052,PA,1,TRAV12N-3*01,TRAJ22*01,CALSPSSGSWQLIF,tgtgctctgagcccatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ2-1*01,CASSLGAEQFF,tgtgctagcagtttaggcgctgagcagttcttc,mouse_tcr0346.clone
mouse_subject0052,PA,1,TRAV4D-4*03,TRAJ15*01,CAAEIQGGRALIF,tgtgctgctgaaatccagggaggcagagctctgatattt,TRBV17*01,TRBJ2-5*01,CASSREDRGPQDTQYF,tgtgctagcagtagagaggacagggggccccaagacacccagtacttt,mouse_tcr0064.clone
mouse_subject0052,PA,1,TRAV12D-2*02,TRAJ16*01,CALSDRPSSGQKLVF,tgtgctttgagtgatcggccttcaagtggccagaagctggttttt,TRBV2*01,TRBJ1-2*01,CASSQEGQANSDYTF,tgtgccagcagccaagagggacaggcaaactccgactacaccttc,mouse_tcr0244.clone
mouse_subject0052,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggttccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLSGFEQYF,tgtgctagcagtttatcagggtttgaacagtacttc,mouse_tcr0377.clone
mouse_subject0052,PA,1,TRAV6D-7*01,TRAJ50*01,CALSVGSSSFSKLVF,tgtgctctgagtgtgggatcctcctccttcagcaagctggtgttt,TRBV29*01,TRBJ2-7*01,CASSTGDEQYF,tgtgctagcagtactggggatgaacagtacttc,mouse_tcr0230.clone
mouse_subject0052,PA,3,TRAV12N-3*01,TRAJ22*01,CALSGTSGSWQLIF,tgtgctctgagtgggacttctggcagctggcaactcatcttt,TRBV19*03,TRBJ2-7*01,CASSLRDWGLYEQYF,tgtgccagcagtctccgggactggggtctctatgaacagtacttc,mouse_tcr0115.clone
mouse_subject0052,PA,1,TRAV6D-6*01,TRAJ26*01,CALGEGNNYAQGLTF,tgtgctctgggtgaggggaataactatgcccagggattaaccttc,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagtttagacaggggggaagtcttcttt,mouse_tcr0286.clone
mouse_subject0052,PA,3,TRAV9-4*01,TRAJ12*01,CALSPTGGYKVVF,tgtgctctgagcccgactggaggctataaagtggtcttt,TRBV29*01,TRBJ2-1*01,CASSWGAEQFF,tgtgctagcagttggggggctgagcagttcttc,mouse_tcr0075.clone
mouse_subject0052,PA,3,TRAV12N-3*01,TRAJ40*01,CALSDTGNYKYVF,tgtgctctgagtgatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-5*01,CASIQYNQAPLF,tgtgctagcatccaatacaaccaggctccgcttttt,mouse_tcr0068.clone
mouse_subject0052,PA,2,TRAV16D/DV11*03,TRAJ53*01,CAMRENSGGSNYKLTF,tgtgctatgagagagaacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-4*01,CASSLSPERLFF,tgtgctagcagtttatctcccgaaagattatttttc,mouse_tcr0055.clone
mouse_subject0052,PA,1,TRAV21/DV12*01,TRAJ53*01,CILNGGSNYKLTF,tgtatcctgaatggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSTGLAPLF,tgtgctagcagtacagggttggctccgcttttt,mouse_tcr0337.clone
mouse_subject0052,PA,9,TRAV21/DV12*01,TRAJ56*01,CILREGATGGNNKLTF,tgtatcctgagagagggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-4*01,CASSLDRGGLFF,tgtgctagcagtctcgacaggggtggattatttttc,mouse_tcr0024.clone
mouse_subject0052,PA,9,TRAV17*02,TRAJ53*01,CALYSGGSNYKLTF,tgtgcactttacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-1*01,CASIGYNYAEQFF,tgtgctagcatcggttataactatgctgagcagttcttc,mouse_tcr0021.clone
mouse_subject0052,PA,3,TRAV12N-3*01,TRAJ12*01,CALSDRGGYKVVF,tgtgctctgagtgatcggggaggctataaagtggtcttt,TRBV29*01,TRBJ2-1*01,CASSSPAEQFF,tgtgctagcagttcacctgctgagcagttcttc,mouse_tcr0145.clone
mouse_subject0052,PA,1,TRAV12N-3*01,TRAJ12*01,CALSWSGGYKVVF,tgtgctctgagctggtctggaggctataaagtggtcttt,TRBV29*01,TRBJ2-5*01,CASSSPDTQYF,tgtgctagcagttccccagacacccagtacttt,mouse_tcr0087.clone
mouse_subject0052,PA,3,TRAV6D-6*01,TRAJ53*01,CALVGGSNYKLTF,tgtgctctggttggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-2*01,CASSPTGQLYF,tgtgctagcagccccaccgggcagctctacttt,mouse_tcr0010.clone
mouse_subject0052,PA,1,TRAV12D-2*02,TRAJ17*01,CALSDYSAGNKLTF,tgtgctttgagtgattacagtgcagggaacaagctaactttt,TRBV15*01,TRBJ1-3*01,CASSADRDGNTLYF,tgtgccagcagtgcggacagggatggaaatacgctctatttt,mouse_tcr0171.clone
mouse_subject0052,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgagggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSSGTEVFF,tgtgctagcagttctggcacagaagtcttcttt,mouse_tcr0048.clone
mouse_subject0052,PA,5,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaaagggagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGSEQYF,tgtgccagcagtatagggtctgaacagtacttc,mouse_tcr0018.clone
mouse_subject0052,PA,1,TRAV6D-6*01,TRAJ31*01,CALAAGSNNRIFF,tgtgctctggccgcggggagcaataacagaatcttcttt,TRBV29*01,TRBJ1-4*01,CASSSGGGLFF,tgtgctagcagttcagggggcggattatttttc,mouse_tcr0027.clone
mouse_subject0052,PA,1,TRAV9-4*01,TRAJ24*01,CALGTTASLGKLQF,tgtgctctggggacaactgccagtttggggaaactgcagttt,TRBV13-2*01,TRBJ2-3*01,CASGDDWASAETLYF,tgtgccagcggtgatgactgggctagtgcagaaacgctgtatttt,mouse_tcr0119.clone
mouse_subject0052,PA,7,TRAV12N-3*01,TRAJ34*02,CALSGTNTNKVVF,tgtgctctgagtggaaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSEGGEQYF,tgtgctagcagtgaagggggagaacagtacttc,mouse_tcr0007.clone
mouse_subject0052,PA,2,TRAV21/DV12*01,TRAJ57*01,CILGNQGGSAKLIF,tgtatcctggggaatcaaggagggtctgcgaagctcatcttt,TRBV29*01,TRBJ1-1*01,CASSFGGEVFF,tgtgctagcagtttcggcggagaagtcttcttt,mouse_tcr0260.clone
mouse_subject0052,PA,2,TRAV12N-3*01,TRAJ22*01,CALSASSGSWQLIF,tgtgctctgagtgcatcttctggcagctggcaactcatcttt,TRBV1*01,TRBJ2-7*01,CTCSAETGEGYEQYF,tgcacctgcagtgcagaaactggggagggatatgaacagtacttc,mouse_tcr0307.clone
mouse_subject0052,PA,1,TRAV6-7/DV9*04,TRAJ56*01,CALSGATGGNNKLTF,tgtgctctgagtggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSYGAEVFF,tgtgctagcagttacggggcagaagtcttcttt,mouse_tcr0310.clone
mouse_subject0052,PA,3,TRAV7-2*01,TRAJ7*01,CAASRGSNNRLTL,tgtgcagcaagcaggggcagcaacaacagacttactttg,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtatcgggggggaagtcttcttt,mouse_tcr0030.clone
mouse_subject0052,PA,1,TRAV12N-3*01,TRAJ40*01,CALSDTGNYKYVF,tgtgctctgagtgatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-1*01,CASSLGAEQFF,tgtgctagcagtttaggggctgagcagttcttc,mouse_tcr0060.clone
mouse_subject0052,PA,8,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtattgggggagaagtcttcttt,mouse_tcr0104.clone
mouse_subject0052,PA,6,TRAV6D-6*01,TRAJ6*01,CALGSGGNYKPTF,tgtgctctgggttcaggaggaaactacaaacctacgttt,TRBV29*01,TRBJ1-5*01,CASSEGEAPLF,tgtgctagctccgagggggaggctccgcttttt,mouse_tcr0003.clone
mouse_subject0052,PA,4,TRAV6-1*02,TRAJ27*01,CVLGYNTNTGKLTF,tgtgttctggggtataacaccaatacaggcaaattaaccttt,TRBV13-3*01,TRBJ2-7*01,CASTGGYEQYF,tgtgccagcactgggggatatgaacagtacttc,mouse_tcr0066.clone
mouse_subject0052,PA,1,TRAV17*02,TRAJ53*01,CARYSGGSNYKLTF,tgtgcacggtacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSWGNEQYF,tgtgctagcagttggggcaatgaacagtacttc,mouse_tcr0013.clone
mouse_subject0004,PA,5,TRAV6D-6*01,TRAJ53*01,CALGRGSNYKLTF,tgtgctctggggagaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASTTGTGVVFF,tgtgctagcaccaccgggacaggggtcgtcttcttt,mouse_tcr0475.clone
mouse_subject0004,PA,1,TRAV13-2*01,TRAJ30*01,CAIPDTNAYKVIF,tgtgctatccctgacacaaatgcttacaaagtcatcttt,TRBV29*01,TRBJ1-2*01,CASSLSSRGQGSDYTF,tgtgctagcagtttatcgagtagggggcagggctccgactacaccttc,mouse_tcr0469.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALEGGSNYKLTF,tgtgctctggaaggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLGGRVFF,tgtgctagcagtttgggggggcgagtcttcttt,mouse_tcr0462.clone
mouse_subject0004,PA,1,TRAV16D/DV11*03,TRAJ45*01,CAMREPNTEGADRLTF,tgtgctatgagagagcccaatacagaaggtgcagatagactcaccttt,TRBV29*01,TRBJ2-1*01,CAAWGENYAEQFF,tgtgctgcctggggggaaaactatgctgagcagttcttc,mouse_tcr0466.clone
mouse_subject0004,PA,1,TRAV4D-3*03,TRAJ53*01,CAADGGGSNYKLTF,tgtgctgctgacggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-1*01,CASSFGAEQFF,tgtgctagcagttttggagctgagcagttcttc,mouse_tcr0775.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ56*01,CALGSGGNNKLTF,tgtgctctgggttctggaggcaataataagctgactttt,TRBV29*01,TRBJ1-5*01,CASSFGQAPLF,tgtgctagcagttttggccaggctccgcttttt,mouse_tcr0459.clone
mouse_subject0004,PA,1,TRAV12N-3*01,TRAJ21*01,CALSEVSNYNVLYF,tgtgctctgagtgaggtgtctaattacaacgtgctttacttc,TRBV19*03,TRBJ1-4*01,CASSPTGGGNERLFF,tgtgccagcagcccgacagggggaggcaacgaaagattatttttc,mouse_tcr0785.clone
mouse_subject0004,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGDEQYF,tgtgccagcagtatcggggatgaacagtacttc,mouse_tcr0783.clone
mouse_subject0004,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSWGGEQYF,tgtgctagcagttggggtggtgaacagtacttc,mouse_tcr0786.clone
mouse_subject0004,PA,2,TRAV21/DV12*01,TRAJ50*01,CILRPGSSSSFSKLVF,tgtatcctgagacctggatcatcctcctccttcagcaagctggtgttt,TRBV29*01,TRBJ2-2*01,CASSLPEGPTGQLYF,tgtgctagcagtttacccgaggggcctaccgggcagctctacttt,mouse_tcr0477.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ40*01,CALGDRRTGNYKYVF,tgtgctctgggtgatcgaaggacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-2*01,CASSLNRGRLYF,tgtgctagcagtttaaacaggggccggctctacttt,mouse_tcr0774.clone
mouse_subject0004,PA,3,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSPDRGRLFF,tgtgctagcagcccggacagggggcgcctcttcttt,mouse_tcr0461.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLGGEVFF,tgtgctagcagtttggggggagaagtcttcttt,mouse_tcr0470.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgaggggagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSLTGEQYF,tgtgctagcagtttaacaggggaacagtacttc,mouse_tcr0479.clone
mouse_subject0004,PA,2,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLGGEQYF,tgtgctagcagtttggggggtgaacagtacttc,mouse_tcr0465.clone
mouse_subject0004,PA,2,TRAV13-1*01,TRAJ45*01,CAPNTEGADRLTF,tgtgctcccaatacagaaggtgcagatagactcaccttt,TRBV29*01,TRBJ1-5*01,CASTQGGAPLF,tgtgctagcacacagggaggggctccgcttttt,mouse_tcr0460.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALGDRGSGGSNYKLTF,tgtgctctgggtgataggggcagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSSGQAPLF,tgtgctagcagttcgggccaggctccgcttttt,mouse_tcr0464.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALGAGSNYKLTF,tgtgctctgggtgcaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLGGEVFF,tgtgctagcagtttggggggagaagtcttcttt,mouse_tcr0468.clone
mouse_subject0004,PA,2,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSSGEAPLF,tgtgctagcagttcaggggaggctccgcttttt,mouse_tcr0455.clone
mouse_subject0004,PA,1,TRAV21/DV12*01,TRAJ53*01,CIRSGGSNYKLTF,tgtatccggagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSWGVEQYF,tgtgctagcagttggggggttgaacagtacttc,mouse_tcr0772.clone
mouse_subject0004,PA,2,TRAV13D-4*01,TRAJ31*01,CVNSNNRIFF,tgtgttaatagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGDEQYF,tgtgccagcagtataggggatgaacagtacttc,mouse_tcr0771.clone
mouse_subject0004,PA,2,TRAV6D-6*03,TRAJ22*01,CALIASSGSWQLIF,tgcgctctgatcgcatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ1-5*01,CASTGGQAPLF,tgtgctagcaccggggggcaggctccgcttttt,mouse_tcr0788.clone
mouse_subject0004,PA,1,TRAV21/DV12*01,TRAJ53*01,CILVGGSNYKLTF,tgtatcctggttggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSSGQAPLF,tgtgctagcagttcgggccaggctccgcttttt,mouse_tcr0773.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-4*01,CASSEGERLFF,tgtgctagcagcgagggcgaaagattatttttc,mouse_tcr0458.clone
mouse_subject0004,PA,1,TRAV6D-6*01,TRAJ53*01,CALAGGSNYKLTF,tgtgctctggccggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-2*01,CASSSSGQLYF,tgtgctagcagttcctcagggcagctctacttt,mouse_tcr0471.clone
mouse_subject0004,PA,3,TRAV6D-6*03,TRAJ22*01,CALIASSGSWQLIF,tgcgctctgatcgcatcttctggcagctggcaactcatcttt,TRBV29*01,TRBJ1-5*01,CASSGGQAPLF,tgtgctagcagcggggggcaggctccgcttttt,mouse_tcr0472.clone
mouse_subject0004,PA,2,TRAV6D-6*01,TRAJ53*01,CALGRGSNYKLTF,tgtgctctggggagaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASTTGTGVVFF,tgtgctagcaccaccgggacaggggttgtcttcttt,mouse_tcr0480.clone
mouse_subject0004,PA,1,TRAV12N-3*01,TRAJ22*01,CALSATSGSWQLIF,tgtgctctgagtgcaacttctggcagctggcaactcatcttt,TRBV19*03,TRBJ2-7*01,CASSPRDWGNYEQYF,tgtgccagcagtcctcgggactgggggaactatgaacagtacttc,mouse_tcr0790.clone
mouse_subject0004,PA,1,TRAV16*01,TRAJ35*01,CAMREGPGFASALTF,tgtgctatgagagagggcccaggctttgcaagtgcgctgacattt,TRBV14*01,TRBJ1-4*01,CASSFNRELSNERLFF,tgtgccagcagtttcaacagggagctttccaacgaaagattatttttc,mouse_tcr0792.clone
mouse_subject0023,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSPDRGEVFF,tgtgctagcagtcccgacaggggggaagtcttcttt,mouse_tcr3071.clone
mouse_subject0023,PA,1,TRAV10*01,TRAJ22*01,CAGSSGSWQLIF,tgtgcaggatcttctggcagctggcaactcatcttt,TRBV5*01,TRBJ2-4*01,CASSPTGGDQNTLYF,tgtgccagcagcccgacagggggcgatcaaaacaccttgtacttt,mouse_tcr3080.clone
mouse_subject0023,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgagggtagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSSGTEVFF,tgtgctagcagctcagggacagaagtcttcttt,mouse_tcr3073.clone
mouse_subject0023,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLGGGVFF,tgtgctagcagtttggggggaggagtcttcttt,mouse_tcr3078.clone
mouse_subject0023,PA,1,TRAV21/DV12*01,TRAJ56*01,CILMEGATGGNNKLTF,tgtatcctgatggaaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ2-7*01,CASSPDRGEQYF,tgtgctagcagccccgacaggggtgaacagtacttc,mouse_tcr3081.clone
mouse_subject0023,PA,1,TRAV9-1*01,TRAJ27*01,CAVSLNTNTGKLTF,tgtgctgtgagccttaacaccaatacaggcaaattaaccttt,TRBV29*01,TRBJ1-2*01,CASSLSNRGQDSDYTF,tgtgctagcagtttatcgaacagggggcaggactccgactacaccttc,mouse_tcr3075.clone
mouse_subject0023,PA,1,TRAV16D/DV11*03,TRAJ48*01,CAMRAPANYGNEKITF,tgtgctatgagagccccggccaactatggaaatgagaaaataactttt,TRBV13-1*01,TRBJ1-4*01,CASSASNFSNERLFF,tgtgccagcagtgcctcaaatttttccaacgaaagattatttttc,mouse_tcr3072.clone
mouse_subject0023,PA,2,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggttccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLGGYEQYF,tgtgctagcagtttaggggggtatgaacagtacttc,mouse_tcr3074.clone
mouse_subject0023,PA,1,TRAV7-5*01,TRAJ17*01,CAVKSAGNKLTF,tgtgcagtgaaaagtgcagggaacaagctaactttt,TRBV29*01,TRBJ2-7*01,CASSLGDEQYF,tgtgctagcagtttaggggatgaacagtacttc,mouse_tcr3082.clone
mouse_subject0023,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSWGSEQYF,tgtgctagcagttgggggagtgaacagtacttc,mouse_tcr3083.clone
mouse_subject0023,PA,1,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaagggcagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGDEQYF,tgtgccagcagtataggggatgaacagtacttc,mouse_tcr3084.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSGGGEQYF,tgtgctagcagtggggggggcgaacagtacttc,mouse_tcr0110.clone
mouse_subject0053,PA,4,TRAV6D-6*01,TRAJ40*01,CALGDRGTGNYKYVF,tgtgctctgggtgatcggggtacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-1*01,CASSLDRGEVFF,tgtgctagcagcctcgacaggggagaagtcttcttt,mouse_tcr0011.clone
mouse_subject0053,PA,1,TRAV7-5*03,TRAJ42*01,CAVSGNSGGSNAKLTF,tgtgcagtgagcgggaattctggaggaagcaatgcaaagctaaccttc,TRBV29*01,TRBJ2-7*01,CASSPDRGEQYF,tgtgctagcagcccggacaggggagaacagtacttc,mouse_tcr0037.clone
mouse_subject0053,PA,1,TRAV12N-3*01,TRAJ34*02,CALSGSNTNKVVF,tgtgctctgagtggctccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-1*01,CASSGGAEQFF,tgtgccagcagtgggggtgctgagcagttcttc,mouse_tcr0031.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ31*01,CALGSGSNNRIFF,tgtgctctgggtagtgggagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSMGNPVFF,tgtgccagcagtatggggaacccagtcttcttt,mouse_tcr0130.clone
mouse_subject0053,PA,1,TRAV9D-4*04,TRAJ45*01,CAVSVRGTEGADRLTF,tgtgctgtgagcgttagggggacagaaggtgcagatagactcaccttt,TRBV2*01,TRBJ2-7*01,CASSQDEDGFYEQYF,tgtgccagcagccaagatgaagatggcttctatgaacagtacttc,mouse_tcr0366.clone
mouse_subject0053,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagttggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRAEVFF,tgtgctagcagcctcgacagggcagaagtcttcttt,mouse_tcr0051.clone
mouse_subject0053,PA,2,TRAV4D-3*03,TRAJ53*01,CAADRGGSNYKLTF,tgtgctgctgaccgtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-1*01,CASSFGGMQFF,tgtgctagcagttttgggggtatgcagttcttc,mouse_tcr0056.clone
mouse_subject0053,PA,3,TRAV13-1*01,TRAJ31*01,CALERWSNNRIFF,tgtgctttggaacgttggagcaataacagaatcttcttt,TRBV17*01,TRBJ1-3*01,CASSDRDRDPGNTLYF,tgtgctagcagtgaccgggacagggatcctggaaatacgctctatttt,mouse_tcr0041.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ17*01,CALGDRGTNSAGNKLTF,tgtgctctgggtgatcgggggactaacagtgcagggaacaagctaactttt,TRBV29*01,TRBJ2-2*01,CASSPDRGQLYF,tgtgctagcagcccggacagggggcagctctacttt,mouse_tcr0182.clone
mouse_subject0053,PA,2,TRAV12D-3*02,TRAJ34*02,CALSGTNTNKVVF,tgtgctttgagtgggaccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSWGDEQYF,tgtgctagcagttggggtgatgaacagtacttc,mouse_tcr0040.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-1*01,CASSTGGGEFF,tgtgctagcagtacagggggcggggagttcttc,mouse_tcr0330.clone
mouse_subject0053,PA,2,TRAV9-4*01,TRAJ27*01,CALNTNTGKLTF,tgtgctcttaacaccaatacaggcaaattaaccttt,TRBV29*01,TRBJ2-7*01,CASWDRGSSYEQYF,tgtgctagctgggacaggggaagctcctatgaacagtacttc,mouse_tcr0120.clone
mouse_subject0053,PA,1,TRAV7-2*01,TRAJ31*01,CAASKGSNNRIFF,tgtgcagcaagcaaggggagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGDEQYF,tgtgccagcagtataggggatgaacagtacttc,mouse_tcr0088.clone
mouse_subject0053,PA,2,TRAV9N-2*01,TRAJ53*01,CVFSGGSNYKLTF,tgtgttttcagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSGGGEVFF,tgtgctagttccggaggaggggaagtcttcttt,mouse_tcr0101.clone
mouse_subject0053,PA,5,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*03,TRBJ2-7*01,CASSWGDEQYF,tgtgccagcagttggggggatgaacagtacttc,mouse_tcr0105.clone
mouse_subject0053,PA,1,TRAV12D-3*02,TRAJ34*02,CALSGTNTNKVVF,tgtgctttgagtgggaccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-7*01,CASTWGGEQYF,tgtgccagcacgtggggaggtgaacagtacttc,mouse_tcr0188.clone
mouse_subject0053,PA,1,TRAV9N-2*01,TRAJ15*01,CVLSARGGRALIF,tgtgttttgagcgcgaggggaggcagagctctgatattt,TRBV17*01,TRBJ2-7*01,CASSREQDRGWEQYF,tgtgctagcagtagagagcaggacaggggatgggaacagtacttc,mouse_tcr0277.clone
mouse_subject0053,PA,1,TRAV9N-2*01,TRAJ33*01,CVLSFNSNYQLIW,tgtgttttgagctttaatagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSWGGEVFF,tgtgctagcagttgggggggagaagtcttcttt,mouse_tcr0374.clone
mouse_subject0053,PA,1,TRAV9N-2*01,TRAJ48*01,CVLSAGGYGNEKITF,tgtgttttgagcgctggtggctatggaaatgagaaaataactttt,TRBV29*01,TRBJ1-4*01,CASSLSGTDFSNERLFF,tgtgctagcagtttatcggggacagatttttccaacgaaagattatttttc,mouse_tcr0268.clone
mouse_subject0053,PA,3,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtatcgggggagaagtcttcttt,mouse_tcr0187.clone
mouse_subject0053,PA,1,TRAV4D-3*03,TRAJ17*01,CAADRTNSAGNKLTF,tgtgctgctgatagaactaacagtgcagggaacaagctaactttt,TRBV13-3*01,TRBJ1-2*01,CASSEGQPNSDYTF,tgtgccagcagtgaaggacaaccaaactccgactacaccttc,mouse_tcr0215.clone
mouse_subject0053,PA,1,TRAV9D-4*04,TRAJ45*01,CAVSARTEGADRLTF,tgtgctgtgagcgcacggacagaaggtgcagatagactcaccttt,TRBV2*01,TRBJ2-7*01,CASSQEADSYEQYF,tgtgccagcagccaagaggcggacagctatgaacagtacttc,mouse_tcr0092.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ33*01,CALGDGGNYQLIW,tgtgctctgggtgacgggggcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSLGSEQYF,tgtgctagcagtttagggtctgaacagtacttc,mouse_tcr0289.clone
mouse_subject0053,PA,1,TRAV17*02,TRAJ57*01,CALEGNQGGSAKLIF,tgtgcactggaggggaatcaaggagggtctgcgaagctcatcttt,TRBV29*01,TRBJ2-1*01,CASSYGAEQFF,tgtgctagcagttacggggctgagcagttcttc,mouse_tcr0352.clone
mouse_subject0053,PA,3,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggaagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGNEQYF,tgtgccagcagtatcgggaatgaacagtacttc,mouse_tcr0174.clone
mouse_subject0053,PA,2,TRAV6-7/DV9*04,TRAJ45*01,CALSDNTEGADRLTF,tgtgctctgagtgataacacagaaggtgcagatagactcaccttt,TRBV29*01,TRBJ2-1*01,CASSFGGMQFF,tgtgctagcagttttgggggtatgcagttcttc,mouse_tcr0255.clone
mouse_subject0053,PA,1,TRAV12D-3*02,TRAJ34*02,CALSGTNTNKVVF,tgtgctttgagtggaaccaataccaacaaagtcgtcttt,TRBV19*01,TRBJ1-4*01,CASSIGERLFF,tgtgccagcagtatcggtgaaagattatttttc,mouse_tcr0077.clone
mouse_subject0053,PA,1,TRAV7-5*03,TRAJ43*01,CAERDNNNAPRF,tgtgctgaacgagataacaacaatgccccacgattt,TRBV29*01,TRBJ2-7*01,CASSGGGEQYF,tgtgctagcagtggggggggcgaacagtacttc,mouse_tcr0086.clone
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLGDEQYF,tgtgctagcagtttaggtgatgaacagtacttc,mouse_tcr0014.clone
mouse_subject0053,PA,1,TRAV12N-3*01,TRAJ34*02,CALSGTNTNKVVF,tgtgctctgagtgggaccaataccaacaaagtcgtcttt,TRBV19*03,TRBJ2-7*01,CASSSPGEQYF,tgtgccagcagttcccccggggagcagtacttc,mouse_tcr0373.clone
mouse_subject0047,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcagagggagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtattgggggagaagtcttcttt,mouse_tcr2617.clone
mouse_subject0047,PA,1,TRAV9N-2*01,TRAJ33*01,CVLSLDSNYQLIW,tgtgttttgagtttggatagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSGGAEVFF,tgtgctagcagtgggggcgcagaagtcttcttt,mouse_tcr2676.clone
mouse_subject0047,PA,1,TRAV12D-3*02,TRAJ12*01,CALSPTGGYKVVF,tgtgctttgagtccgactggaggctataaagtggtcttt,TRBV29*01,TRBJ2-7*01,CASTEGGEQYF,tgtgctagcaccgaggggggggaacagtacttc,mouse_tcr2687.clone
mouse_subject0047,PA,1,TRAV9N-3*01,TRAJ26*01,CAVMSNYAQGLTF,tgtgctgtgatgagtaactatgcccagggattaaccttc,TRBV19*01,TRBJ2-1*01,CASSIGSEQFF,tgtgccagcagtattgggagtgagcagttcttc,mouse_tcr2739.clone
mouse_subject0047,PA,2,TRAV7-5*01,TRAJ23*01,CAVSLNYNQGKLIF,tgtgcagtgagcctcaattataaccaggggaagcttatcttt,TRBV29*01,TRBJ2-7*01,CASSGGNEQYF,tgtgctagcagtggagggaatgaacagtacttc,mouse_tcr2675.clone
mouse_subject0047,PA,1,TRAV5-4*01,TRAJ40*01,CAASNTGNYKYVF,tgtgctgcaagtaatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ1-1*01,CASTLGGEVFF,tgtgctagcaccctagggggcgaagtcttcttt,mouse_tcr2714.clone
mouse_subject0047,PA,1,TRAV6D-6*03,TRAJ53*01,CALGGGSNYKLTF,tgcgctttagggggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSSGGGVFF,tgtgctagcagctcagggggaggagtcttcttt,mouse_tcr2631.clone
mouse_subject0047,PA,2,TRAV9N-2*01,TRAJ34*02,CVLSAGNTNKVVF,tgtgttttgagcgcggggaataccaacaaagtcgtcttt,TRBV29*01,TRBJ1-4*01,CASSFGERLFF,tgtgctagcagtttcggcgaaagattatttttc,mouse_tcr2664.clone
mouse_subject0047,PA,3,TRAV9-1*01,TRAJ27*01,CAVSHNTNTGKLTF,tgtgctgtgagccataacaccaatacaggcaaattaaccttt,TRBV29*01,TRBJ1-2*01,CASSLSSRGPNSDYTF,tgtgctagcagtttatcgtctagggggcccaactccgactacaccttc,mouse_tcr2622.clone
mouse_subject0022,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSRGSNYKLTF,tgtatcctctcccgaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSQGREQYF,tgtgctagcagtcagggccgagaacagtacttc,mouse_tcr2541.clone
mouse_subject0022,PA,1,TRAV6-2*01,TRAJ33*01,CVLGPSSNYQLIW,tgtgttctgggtccgtctagcaactatcagttgatctgg,TRBV29*01,TRBJ1-5*01,CASSFGQAPLF,tgtgctagcagttttggccaggctccgcttttt,mouse_tcr2540.clone
mouse_subject0022,PA,1,TRAV6D-6*01,TRAJ40*01,CALGDRVTGNYKYVF,tgtgctctgggtgatcgggttacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-7*01,CASSWDRGEQYF,tgtgctagcagttgggacaggggtgaacagtacttc,mouse_tcr2542.clone
mouse_subject0022,PA,1,TRAV9-1*01,TRAJ15*01,CAVSAGGGRALIF,tgtgctgtgagcgcggggggaggcagagctctgatattt,TRBV13-3*01,TRBJ1-4*01,CASSGTGGHERLFF,tgtgccagcagtgggacagggggccacgaaagattatttttc,mouse_tcr2536.clone
mouse_subject0022,PA,2,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSEGREQYF,tgtgctagcagtgaggggcgcgaacagtacttc,mouse_tcr2539.clone
mouse_subject0022,PA,1,TRAV9-1*01,TRAJ4*01,CAVSLSGSFNKLTF,tgtgctgtgagcttatctggtagcttcaataagttgaccttt,TRBV13-3*01,TRBJ2-4*01,CASSDAGGQNTLYF,tgtgccagcagtgatgcagggggtcaaaacaccttgtacttt,mouse_tcr2537.clone
mouse_subject0022,PA,2,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggttccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLGGYEQYF,tgtgctagcagtttagggggttatgaacagtacttc,mouse_tcr2535.clone
mouse_subject0005,PA,2,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagccgcggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGSEQYF,tgtgccagcagtatcggatcggaacagtacttc,mouse_tcr0392.clone
mouse_subject0005,PA,1,TRAV12N-3*01,TRAJ28*01,CALSGTGSNRLTF,tgtgctctctcaggcactgggagtaacaggctcactttt,TRBV19*01,TRBJ1-4*01,CASSIGERLFF,tgtgccagcagtattggcgaaagattatttttc,mouse_tcr0394.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ42*01,CALHSGGSNAKLTF,tgtgctctccattctggaggaagcaatgcaaagctaaccttc,TRBV29*01,TRBJ1-4*01,CASTMGERLFF,tgtgctagcaccatgggcgaaagattatttttc,mouse_tcr0401.clone
mouse_subject0005,PA,1,TRAV9-4*01,TRAJ45*01,CALSMRTEGADRLTF,tgtgctctgagcatgaggacagaaggtgcagatagactcaccttt,TRBV2*01,TRBJ2-7*01,CASSQEGDSYEQYF,tgtgccagcagccaagagggggactcctatgaacagtacttc,mouse_tcr0386.clone
mouse_subject0005,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggagctactggaggcaataataagctgactttt,TRBV29*01,TRBJ2-2*01,CASSHDRGQLYF,tgtgctagcagccacgacagggggcagctctacttt,mouse_tcr0406.clone
mouse_subject0005,PA,1,TRAV12-1*01,TRAJ16*01,CALSDRPSSGQKLVF,tgtgctttgagtgatcggccttcaagtggccagaagctggttttt,TRBV2*01,TRBJ1-2*01,CASSQEGGPNSDYTF,tgtgccagcagccaagaggggggcccaaactccgactacaccttc,mouse_tcr0799.clone
mouse_subject0005,PA,1,TRAV12N-3*01,TRAJ40*01,CALSYTGNYKYVF,tgtgctctgagttatacaggaaactacaaatacgtcttt,TRBV29*01,TRBJ2-5*01,CASSQGDTQYF,tgtgctagcagccaaggggacacccagtacttt,mouse_tcr0396.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgagggtagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcctatgaacagtacttc,mouse_tcr0393.clone
mouse_subject0005,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSLDRGQVFF,tgtgctagcagtttagacagggggcaagtcttcttt,mouse_tcr0400.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-4*01,CASSLGERLFF,tgtgctagcagtttaggcgaaagattatttttc,mouse_tcr0390.clone
mouse_subject0005,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagccgcggtagcaataacagaatcttcttt,TRBV29*01,TRBJ2-7*01,CASSLGGYEQYF,tgtgctagcagtttggggggttatgaacagtacttc,mouse_tcr0804.clone
mouse_subject0005,PA,3,TRAV21/DV12*01,TRAJ53*01,CILNGGSNYKLTF,tgtatcctgaatggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-5*01,CASSFGQAPLF,tgtgctagcagtttcggccaggctccgcttttt,mouse_tcr0409.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASTSGTGVVFF,tgtgctagcacctccgggacaggggtagtcttcttt,mouse_tcr0397.clone
mouse_subject0005,PA,4,TRAV3-4*01,TRAJ56*01,CAVSGGATGGNNKLTF,tgcgcagtcagtggtggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ2-4*01,CASSPDRGALYF,tgtgctagcagccccgacaggggggccttgtacttt,mouse_tcr0388.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgaaggtagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcttatgaacagtacttc,mouse_tcr0403.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ40*01,CALGAGNTGNYKYVF,tgtgctctgggtgctggtaatacaggaaactacaaatacgtcttt,TRBV14*01,TRBJ2-3*01,CASSGDWGGETLYF,tgtgccagcagtggggactggggcggcgaaacgctgtatttt,mouse_tcr0802.clone
mouse_subject0005,PA,1,TRAV9N-2*01,TRAJ34*02,CVLSPSNTNKVVF,tgtgttttgagcccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ1-4*01,CASSLGERLFF,tgtgctagcagtttaggcgaaagattatttttc,mouse_tcr0389.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ33*01,CALGAGSNYQLIW,tgtgctctgggggccgggagcaactatcagttgatctgg,TRBV29*01,TRBJ2-3*01,CASSSGETLYF,tgtgctagcagttcaggggaaacgctgtatttt,mouse_tcr0399.clone
mouse_subject0005,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggagctactggaggcaataataagctgactttt,TRBV29*01,TRBJ2-4*01,CASSPDRGALYF,tgtgctagcagccccgacaggggggccttgtacttt,mouse_tcr0793.clone
mouse_subject0005,PA,1,TRAV12N-3*01,TRAJ37*01,CALTGNTGKLIF,tgtgctctgacaggcaataccggaaaactcatcttt,TRBV2*01,TRBJ2-7*01,CASSQETGGVTYEQYF,tgtgccagcagccaagaaactgggggggtgacatatgaacagtacttc,mouse_tcr0800.clone
mouse_subject0005,PA,1,TRAV6D-6*01,TRAJ42*01,CALHSGGSNAKLTF,tgtgctctccattctggaggaagcaatgcaaagctaaccttc,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcctatgaacagtacttc,mouse_tcr0797.clone
mouse_subject0005,PA,1,TRAV6-7/DV9*02,TRAJ15*01,CALGAQGGRALIF,tgtgctctgggtgcccagggaggcagagctctgatattt,TRBV29*01,TRBJ2-2*01,CASSHDRGQLYF,tgtgctagcagccacgacagggggcagctctacttt,mouse_tcr0796.clone
mouse_subject0005,PA,1,TRAV3-1*01,TRAJ56*01,CAVSGGATGGNNKLTF,tgcgcagtcagtggtggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ2-4*01,CASSPDRGALYF,tgtgctagcagccccgacaggggggccttgtacttt,mouse_tcr0806.clone
mouse_subject0005,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggaagcaataacagaatcttcttt,TRBV19*01,TRBJ1-1*01,CASSIGGEVFF,tgtgccagcagtatagggggagaagtcttcttt,mouse_tcr0408.clone
mouse_subject0005,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGGEQYF,tgtgccagcagtattgggggcgaacagtacttc,mouse_tcr0395.clone
mouse_subject0005,PA,1,TRAV9-4*01,TRAJ23*01,CALTSNQGKLIF,tgtgctctgacctctaaccaggggaagcttatcttt,TRBV13-1*01,TRBJ2-3*01,CASSDADWGVGETLYF,tgtgccagcagtgatgcagactggggggtaggagaaacgctgtatttt,mouse_tcr0404.clone
mouse_subject0005,PA,2,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggtcccttccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLGGYEQYF,tgtgctagcagtttaggggggtatgaacagtacttc,mouse_tcr0387.clone
mouse_subject0005,PA,1,TRAV12-1*01,TRAJ28*01,CALTDHQPGTGSNRLTF,tgtgctttgaccgaccatcaaccaggcactgggagtaacaggctcactttt,TRBV29*01,TRBJ2-3*01,CASSSGETLYF,tgtgctagcagttcaggggaaacgctgtatttt,mouse_tcr0805.clone
mouse_subject0005,PA,1,TRAV9N-2*01,TRAJ15*01,CVLSAWGGRALIF,tgtgttttgagcgcgtggggaggcagagctctgatattt,TRBV17*01,TRBJ2-7*01,CASSRETGVSYEQYF,tgtgctagcagtagagagactggggtctcctatgaacagtacttc,mouse_tcr0391.clone
mouse_subject0005,PA,1,TRAV21/DV12*01,TRAJ53*01,CILSGGSNYKLTF,tgtatcctgagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-1*01,CASSLGAEQFF,tgtgctagcagtttaggagctgagcagttcttc,mouse_tcr0795.clone
mouse_subject0005,PA,1,TRAV12N-3*01,TRAJ22*01,CALSPSSGSWQLIF,tgtgctctgagtccatcttctggcagctggcaactcatcttt,TRBV13-2*01,TRBJ2-7*01,CASGDRDWEVYEQYF,tgtgccagcggtgatagggactgggaggtctatgaacagtacttc,mouse_tcr0803.clone
mouse_subject0005,PA,1,TRAV6-7/DV9*02,TRAJ15*01,CALGAQGGRALIF,tgtgctctgggtgcccagggaggcagagctctgatattt,TRBV13-1*01,TRBJ2-1*01,CASSGNAEQFF,tgtgccagcagtggaaatgctgagcagttcttc,mouse_tcr0794.clone
mouse_subject0020,PA,1,TRAV6-4*01,TRAJ34*02,CALVPSNTNKVVF,tgtgctctggttccctccaataccaacaaagtcgtcttt,TRBV29*01,TRBJ2-7*01,CASSLSGYEQYF,tgtgctagcagtttatcagggtatgaacagtacttc,mouse_tcr1539.clone
mouse_subject0020,PA,2,TRAV12D-3*02,TRAJ58*01,CALSDGGTGSKLSF,tgtgctttgagtgatgggggcactgggtctaagctgtcattt,TRBV29*01,TRBJ2-7*01,CASSPDRGEQYF,tgtgctagcagtccggacagaggggaacagtacttc,mouse_tcr1707.clone
mouse_subject0020,PA,1,TRAV6D-6*01,TRAJ33*01,CALGEGSNYQLIW,tgtgctctgggtgaagggagcaactatcagttgatctgg,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcttatgaacagtacttc,mouse_tcr1571.clone
mouse_subject0020,PA,1,TRAV6D-6*01,TRAJ53*01,CALEGGSNYKLTF,tgtgctctggagggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-4*01,CASSGGSTLYF,tgtgctagcagtggggggagcaccttgtacttt,mouse_tcr1656.clone
mouse_subject0020,PA,1,TRAV21/DV12*01,TRAJ53*01,CILLGGSNYKLTF,tgtatccttcttggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLFGEQYF,tgtgctagcagtttatttggggaacagtacttc,mouse_tcr1567.clone
mouse_subject0020,PA,1,TRAV21/DV12*01,TRAJ56*01,CILRAGATGGNNKLTF,tgtatcctgagagccggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-4*01,CASSPDRGRLFF,tgtgctagcagtcctgacaggggaagattatttttc,mouse_tcr1683.clone
mouse_subject0020,PA,1,TRAV7-2*01,TRAJ31*01,CAASRGSNNRIFF,tgtgcagcaagcaggggtagcaataacagaatcttcttt,TRBV19*01,TRBJ2-7*01,CASSIGGEQYF,tgtgccagcagtatcgggggggaacagtacttc,mouse_tcr1587.clone
mouse_subject0020,PA,1,TRAV12N-3*01,TRAJ22*01,CALSATSGSWQLIF,tgtgctctgagtgcgacttctggcagctggcaactcatcttt,TRBV19*03,TRBJ2-1*01,CASSLRDWENAEQFF,tgtgccagcagcctccgggactgggagaatgctgagcagttcttc,mouse_tcr1617.clone
mouse_subject0020,PA,5,TRAV21/DV12*01,TRAJ56*01,CILRVGATGGNNKLTF,tgtatcctgagagtaggggctactggaggcaataataagctgactttt,TRBV29*01,TRBJ1-1*01,CASSPDRGQVFF,tgtgctagcagtccagacagggggcaagtcttcttt,mouse_tcr1547.clone
mouse_subject0020,PA,1,TRAV8-2*01,TRAJ12*01,CAKTGGYKVVF,tgtgctaagactggaggctataaagtggtcttt,TRBV13-3*01,TRBJ2-7*01,CASSERDREDEQYF,tgtgccagcagtgaaagggacagggaggacgaacagtacttc,mouse_tcr1659.clone
mouse_subject0020,PA,4,TRAV6-2*01,TRAJ27*01,CVLGYNTNTGKLTF,tgtgttctgggctataacaccaatacaggcaaattaaccttt,TRBV13-3*01,TRBJ2-7*01,CASTGGYEQYF,tgtgccagcactgggggctatgaacagtacttc,mouse_tcr1615.clone
mouse_subject0020,PA,1,TRAV6-5*01,TRAJ31*01,CALGDRYSNNRIFF,tgtgctctgggtgatcgatatagcaataacagaatcttcttt,TRBV19*03,TRBJ2-7*01,CGSAGGAEQYF,tgtggcagcgcggggggggctgaacagtacttc,mouse_tcr1608.clone
mouse_subject0020,PA,1,TRAV6D-6*01,TRAJ33*01,CALAPRGSNYQLIW,tgtgctctggccccccggggcagcaactatcagttgatctgg,TRBV29*01,TRBJ1-5*01,CASWDSNQAPLF,tgtgctagctgggacagcaaccaggctccgcttttt,mouse_tcr1657.clone
mouse_subject0020,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcatatgaacagtacttc,mouse_tcr1709.clone
mouse_subject0020,PA,1,TRAV9-1*01,TRAJ12*01,CAVSAGGGYKVVF,tgtgctgtgagcgcgggtggaggctataaagtggtcttt,TRBV13-3*01,TRBJ1-2*01,CASSETGGHSDYTF,tgtgccagcagtgaaacagggggccactccgactacaccttc,mouse_tcr1582.clone
mouse_subject0020,PA,2,TRAV6D-6*01,TRAJ53*01,CALAGGSNYKLTF,tgtgctctggcgggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSSYEQYF,tgtgctagcagttcatatgaacagtacttc,mouse_tcr1645.clone
mouse_subject0020,PA,2,TRAV6D-6*03,TRAJ53*01,CALCGGSNYKLTF,tgcgctctgtgtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSLGGEQYF,tgtgctagcagtttggggggtgaacagtacttc,mouse_tcr1605.clone
mouse_subject0020,PA,3,TRAV12N-3*01,TRAJ34*02,CALSASNTNKVVF,tgtgctctgagtgcctccaataccaacaaagtcgtcttt,TRBV19*01,TRBJ2-1*01,CASSTGAEQFF,tgtgccagcagtacgggggctgagcagttcttc,mouse_tcr1664.clone
mouse_subject0020,PA,2,TRAV9N-2*01,TRAJ31*01,CVLSARNNRIFF,tgtgttttgagcgcgcgcaataacagaatcttcttt,TRBV19*01,TRBJ2-3*01,CASSISLRQFSAETLYF,tgtgccagcagtatatccctccgacaatttagtgcagaaacgctgtatttt,mouse_tcr1586.clone
mouse_subject0020,PA,1,TRAV4D-3*03,TRAJ53*01,CAAYSGGSNYKLTF,tgtgctgcttacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSFGGEVFF,tgtgctagcagttttgggggggaagtcttcttt,mouse_tcr1704.clone
mouse_subject0020,PA,1,TRAV6D-6*01,TRAJ33*01,CALGRGSNYQLIW,tgtgctctggggaggggaagcaactatcagttgatctgg,TRBV29*01,TRBJ1-1*01,CASSTGSEVFF,tgtgctagcagtacagggtcagaagtcttcttt,mouse_tcr1668.clone
mouse_subject0020,PA,1,TRAV21/DV12*01,TRAJ53*01,CIPLGGSNYKLTF,tgtatccccctcggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-4*01,CASSAPNTLYF,tgtgctagcagtgccccgaacaccttgtacttt,mouse_tcr1658.clone
mouse_subject0045,PA,1,TRAV6D-6*01,TRAJ22*01,CALGSGGSWQLIF,tgtgctctggggtccgggggcagctggcaactcatcttt,TRBV29*01,TRBJ1-1*01,CASSGPEVFF,tgtgctagcagtggcccagaagtcttcttt,mouse_tcr1919.clone
mouse_subject0045,PA,1,TRAV16D/DV11*03,TRAJ53*01,CAMRGNSGGSNYKLTF,tgtgctatgagggggaacagtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ1-1*01,CASSLFPTEVFF,tgtgctagcagtttattccccacagaagtcttcttt,mouse_tcr2100.clone
"""

        filename = path / 'data.csv'

        with open(filename, "w") as file:
            file.writelines(data)

        dataset = SingleLineReceptorImport.import_dataset({
            "path": filename,
            "result_path": path / "result/",
            "separator": ",",
            "columns_to_load": ["subject", "epitope", "count", "v_a_gene", "j_a_gene", "cdr3_a_aa", "v_b_gene", "j_b_gene", "cdr3_b_aa", "clone_id"],
            "column_mapping": {
                "cdr3_a_aa": "alpha_amino_acid_sequence",
                "cdr3_b_aa": "beta_amino_acid_sequence",
                "v_a_gene": "alpha_v_gene",
                "v_b_gene": "beta_v_gene",
                "j_a_gene": "alpha_j_gene",
                "j_b_gene": "beta_j_gene",
                "clone_id": "identifier"
            },
            "receptor_chains": "TRA_TRB",
            "region_type": "IMGT_CDR3",
            "sequence_file_size": 50000,
            "organism": "mouse"
        }, "dataset name 2")

        self.assertEqual(324, dataset.get_example_count())
        self.assertTrue(all(item.identifier is not None for item in dataset.get_data()))
        self.assertTrue(os.path.isfile(path / "result/batch1.npy"))
        self.assertTrue(os.path.isfile(path / "result/dataset name 2.iml_dataset"))
        self.assertEqual("mouse", dataset.labels["organism"])

        shutil.rmtree(path)
