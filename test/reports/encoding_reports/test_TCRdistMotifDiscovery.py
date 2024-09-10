import os
import shutil

import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ml_reports.TCRdistMotifDiscovery import TCRdistMotifDiscovery
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def _create_dataset(path):
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
mouse_subject0053,PA,1,TRAV6D-6*01,TRAJ53*01,CALGGGSNYKLTF,tgtgctctgggtggaggcagcaattacaaactgacattt,TRBV29*01,TRBJ2-7*01,CASSGGGEQYF,tgtgctagcagtggggggggcgaacagtacttc,mouse_tcr0110.clone"""

    receptors = []
    for line in data.split("\n"):
        if not line.startswith('subject,epitope'):
            receptor_info = line.split(",")
            receptor = TCABReceptor(alpha=ReceptorSequence(sequence_aa=receptor_info[5], sequence=receptor_info[6],
                                                           metadata=SequenceMetadata(v_call=receptor_info[3], locus='TRA',
                                                                                     j_call=receptor_info[4])),
                                    beta=ReceptorSequence(sequence_aa=receptor_info[9], sequence=receptor_info[10],
                                                          metadata=SequenceMetadata(v_call=receptor_info[7], locus='TRB',
                                                                                    j_call=receptor_info[9])),
                                    metadata={'epitope': receptor_info[1]})
            receptors.append(receptor)

    return ReceptorDataset.build_from_objects(receptors, 100, path, labels={'epitope': ['PA'], 'organism': 'mouse'})


def test_generate():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "tcrdist_motif_discovery/")
    dataset = _create_dataset(PathBuilder.build(path / 'dataset'))

    dataset = TCRdistEncoder(8).encode(dataset, EncoderParams(path / "result", LabelConfiguration([Label("epitope", None)])))

    report = TCRdistMotifDiscovery(train_dataset=dataset, test_dataset=dataset, result_path=path / "report",
                                   name="report name", cores=8, positive_class_name="PA", min_cluster_size=3)
    report.label = Label("epitope", None)
    report._generate()

    shutil.rmtree(path)
