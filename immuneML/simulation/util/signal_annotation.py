from typing import List

import pandas as pd
import bionumpy as bnp

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.bnp_util import get_sequence_field_name
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.util import annotate_sequences, make_annotated_dataclass


def annotate_sequence_dataset(dataset: SequenceDataset, signals: List[Signal],
                              region_type: RegionType = RegionType.IMGT_CDR3,
                              sequence_type: SequenceType = SequenceType.AMINO_ACID) -> pd.DataFrame:
    data = dataset.data
    sequences = BackgroundSequences(sequence_aa=getattr(data, get_sequence_field_name(region_type,
                                                                            SequenceType.AMINO_ACID)),
                                    sequence=getattr(data,
                                                     get_sequence_field_name(region_type, SequenceType.NUCLEOTIDE)),
                                    v_call=data.v_call, j_call=data.j_call,
                                    region_type=[region_type.value for _ in range(len(data))],
                                    frame_type=['' for _ in range(len(data))], p_gen=[-1 for _ in range(len(data))],
                                    duplicate_count=[1 for _ in range(len(data))], locus=data.locus,
                                    from_default_model=[0 for _ in range(len(data))])

    annotated_dataclass = make_annotated_dataclass([(signal.id, int) for signal in signals] +
                                                   [(f'{signal.id}_positions', str) for signal in signals] +
                                                   [('signals_aggregated', str)], signals)

    seqs_with_signal_matrix = annotate_sequences(sequences=sequences,
                                                 is_amino_acid=sequence_type == SequenceType.AMINO_ACID,
                                                 all_signals=signals,
                                                 annotated_dc=annotated_dataclass,
                                                 region_type=region_type)

    annotated_sequences = seqs_with_signal_matrix.topandas()
    annotated_sequences.drop(columns=['p_gen', 'frame_type', 'region_type', 'duplicate_count', 'from_default_model'],
                             inplace=True)
    annotated_sequences.rename(columns={'sequence_aa': get_sequence_field_name(region_type, SequenceType.AMINO_ACID),
                                        'sequence': get_sequence_field_name(region_type, SequenceType.NUCLEOTIDE)},
                               inplace=True)
    return annotated_sequences
