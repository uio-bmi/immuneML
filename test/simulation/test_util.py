import shutil
from pathlib import Path

from immuneML.data_model.bnp_util import bnp_read_from_file
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.implants.LigoPWM import LigoPWM
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.util import make_annotated_dataclass, annotate_sequences
from immuneML.util.PathBuilder import PathBuilder


def get_sequences(path: Path):
    content = """sequence_aa	sequence	v_call	j_call	region_type	frame_type	p_gen	from_default_model	duplicate_count	locus
CAITQLPGTSGRLGELFF	TGTGCCATCACCCAGCTACCCGGGACTAGCGGGAGGCTCGGGGAGCTGTTTTTT	TRBV10-3*02	TRBJ2-2*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASIICFWDTQYF	TGTGCCAGCATCATCTGCTTTTGGGATACGCAGTATTTT	TRBV28*01	TRBJ2-3*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASRYPSRVSNQPQHF	TGTGCCAGCAGGTATCCTAGCAGGGTAAGCAATCAGCCCCAGCATTTT	TRBV12-3*01	TRBJ1-5*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASSESGQGANVLTF	TGTGCCAGCAGTGAATCGGGACAGGGGGCCAACGTCCTGACTTTC	TRBV25-1*01	TRBJ2-6*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASSAHAPRGSRYQETQYF	TGCGCCAGCAGCGCCCACGCCCCGCGCGGGAGCCGATATCAGGAGACCCAGTACTTC	TRBV5-1*01	TRBJ2-5*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CSASPAMNTEAFF	TGCAGTGCTAGCCCAGCTATGAACACTGAAGCTTTCTTT	TRBV20-1*01	TRBJ1-1*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASSQGLDQTQYF	TGTGCCAGCAGCCAAGGTCTCGACCAGACCCAGTACTTC	TRBV3-1*02	TRBJ2-5*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASRIARGRRGQPQHF	TGTGCCAGCAGAATCGCCCGAGGTCGAAGGGGTCAGCCCCAGCATTTT	TRBV6-5*01	TRBJ1-5*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CASSWRGGLNNEQFF	TGTGCCAGCAGCTGGAGAGGGGGCTTAAACAATGAGCAGTTCTTC	TRBV11-3*01	TRBJ2-1*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CATRGGEPGNTEAFF	TGTGCCACACGGGGGGGCGAGCCTGGGAACACTGAAGCTTTCTTT	TRBV6-4*01	TRBJ1-1*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CSAWDSVLHF	TGCAGTGCTTGGGACAGCGTTCTCCACTTT	TRBV20-1*01	TRBJ1-6*02	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
CARDYEQYF	TGTGCCAGAGACTACGAGCAGTACTTC	TRBV19*01	TRBJ2-7*01	IMGT_JUNCTION	IN	-1.0	1	-1	TRB
"""

    with path.open('w') as file:
        file.write(content)

    return bnp_read_from_file(path, dataclass=BackgroundSequences)


def get_motif(path: Path):

    with path.open('w') as file:
        file.write("""> motif1
A [  0 69  0  0  4  2  7  5  0  0  0  0  0 ]
C [ 69  0  0  0  0  0  0  0  0  0  0  0  0 ]
D [  0  0  0  0  0 23 18  2  0  0  0  0  0 ]
E [  0  0  0  0  5  7  6  1  0 69  0  0  0 ]
F [  0  0  0  0  2  0  0  2  0  0  0  0 69 ]
G [  0  0  0  1  0 19  9  9  0  0  0  0  0 ]
H [  0  0  0  0  1  0  2  2  0  0  0  0  0 ]
I [  0  0  0  0  0  0  0  1  0  0  0  0  0 ]
K [  0  0  0  0  1  0  0  0  0  0  0  0  0 ]
L [  0  0  0  0 32  2  2  0  0  0  0  0  0 ]
M [  0  0  0  1  0  0  1  0  0  0  0  0  0 ]
N [  0  0  0  0  0  0  0  1  0  0  0  0  0 ]
P [  0  0  0  0  7  6  2  9  0  0  0  0  0 ]
Q [  0  0  0  0  3  2  5  0  0  0 69  0  0 ]
R [  0  0  0  0  5  4  2  1  0  0  0  0  0 ]
S [  0  0 69 66  5  1 15 32  0  0  0  0  0 ]
T [  0  0  0  1  0  2  0  4  0  0  0  0  0 ]
V [  0  0  0  0  1  1  0  0  0  0  0  0  0 ]
W [  0  0  0  0  1  0  0  0  1  0  0  0  0 ]
Y [  0  0  0  0  2  0  0  0 68  0  0 69  0 ]
""")

    return LigoPWM.build('motif1', path, 2)


def test_annotate_sequences():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'annotate_sequences')
    sequences = get_sequences(path / 'sequences.tsv')
    motif = get_motif(path / 'motif.jaspar')

    signals = [Signal(id='signal1', motifs=[motif], sequence_position_weights={'104': 0.})]

    dc = make_annotated_dataclass(annotation_fields=sorted(
            [('signal1', int), ('signals_aggregated', str), (f"signal1_positions", str)]), signals=signals)

    annotate_sequences(sequences, is_amino_acid=True, all_signals=signals, annotated_dc=dc, sim_item_name='sim_item')

    shutil.rmtree(path)
