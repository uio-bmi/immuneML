import logging
from dataclasses import make_dataclass
from itertools import chain
from pathlib import Path

import bionumpy as bnp
import numpy as np
from bionumpy import AminoAcidEncoding, DNAEncoding
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encodings import BaseEncoding
from bionumpy.io import delimited_buffers
from bionumpy.sequence.string_matcher import RegexMatcher, StringMatcher
from npstructures import RaggedArray

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.util.PositionHelper import PositionHelper


def get_signal_sequence_count(repertoire_count: int, sim_item) -> int:
    return round(sim_item.receptors_in_repertoire_count * sim_item.repertoire_implanting_rate) * repertoire_count


def get_sequence_per_signal_count(sim_item) -> dict:
    if sim_item.receptors_in_repertoire_count:
        sequence_count = sim_item.receptors_in_repertoire_count * sim_item.number_of_examples
        seq_with_signal_count = {signal.id: get_signal_sequence_count(repertoire_count=sim_item.number_of_examples, sim_item=sim_item)
                                 for signal in sim_item.signals}
        seq_without_signal_count = {'no_signal': sequence_count - sum(seq_with_signal_count.values())}

        return {**seq_with_signal_count, **seq_without_signal_count}
    elif len(sim_item.signals) == 1:
        return {sim_item.signals[0].id: sim_item.number_of_examples, 'no_signal': 0}
    elif len(sim_item.signals) == 0:
        return {'no_signal': sim_item.number_of_examples}
    else:
        raise NotImplementedError


def get_bnp_data(sequence_path, columns_with_types: list = None):
    if columns_with_types is None or isinstance(columns_with_types, list) and len(columns_with_types) == 0:
        data_class = GenModelAsTSV
    else:
        data_class = GenModelAsTSV.extend(tuple(columns_with_types))

    buff_type = delimited_buffers.get_bufferclass_for_datatype(data_class, delimiter='\t', has_header=True)

    with bnp.open(sequence_path, buffer_type=buff_type) as file:
        data = file.read()

    return data


def write_bnp_data(path: Path, data, append_if_exists: bool = True):
    buff_type = delimited_buffers.get_bufferclass_for_datatype(type(data), delimiter="\t", has_header=True)

    if path.is_file() and append_if_exists:
        with bnp.open(path, buffer_type=buff_type, mode='a') as file:
            file.write(data)
    elif not path.is_file():
        with bnp.open(path, buffer_type=buff_type, mode='w') as file:
            file.write(data)
    else:
        raise RuntimeError(f"Tried writing to {path}, but it already exists and append_if_exists parameter is set to False.")


def annotate_sequences(sequences, is_amino_acid: bool, all_signals: list):
    encoding = AminoAcidEncoding if is_amino_acid else DNAEncoding
    sequence_array = sequences.sequence_aa if is_amino_acid else sequences.sequence

    signal_matrix = np.zeros((len(sequence_array), len(all_signals)))
    signal_positions = {}

    for index, signal in enumerate(all_signals):
        signal_pos_col = None
        for motifs, v_call, j_call in signal.get_all_motif_instances(SequenceType.AMINO_ACID if is_amino_acid else SequenceType.NUCLEOTIDE):
            matches_gene = match_genes(v_call, sequences.v_call, j_call, sequences.j_call)
            matches = None

            for motif in motifs:

                matches_motif = match_motif(motif, encoding, sequence_array)
                if matches is None:
                    matches = np.logical_and(matches_motif, matches_gene)
                else:
                    matches = np.logical_or(matches, np.logical_and((matches_motif, matches_gene)))

            signal_pos_col = np.logical_or(signal_pos_col, matches) if signal_pos_col is not None else matches
            signal_matrix[:, index] = np.logical_or(signal_matrix[:, index], np.logical_or.reduce(matches, axis=1))

        np_mask = RaggedArray(np.where(signal_pos_col.ravel(), "1", "0"), shape=signal_pos_col.shape)
        signal_positions[f'{signal.id}_positions'] = ['m' + "".join(np_mask[ind, :]) for ind in range(len(signal_pos_col))]

    signal_matrix = make_bnp_annotated_sequences(sequences, all_signals, signal_matrix, signal_positions)

    return signal_matrix


def match_genes(v_call, v_call_array, j_call, j_call_array):
    if v_call is not None and v_call != "":
        matcher = StringMatcher(v_call, encoding=BaseEncoding)
        matches_gene = matcher.rolling_window(v_call_array, mode='same').any(axis=1).reshape(-1, 1)
    else:
        matches_gene = np.ones(len(v_call_array)).reshape(-1, 1)

    if j_call is not None and j_call != "":
        matcher = StringMatcher(j_call, encoding=BaseEncoding)
        matches_j = matcher.rolling_window(j_call_array, mode='same').any(axis=1).reshape(-1, 1)
        matches_gene = np.logical_and(matches_gene, matches_j)

    return matches_gene.astype(bool)


def match_motif(motif: str, encoding, sequence_array):
    matcher = RegexMatcher(motif, encoding=encoding)
    matches = matcher.rolling_window(sequence_array, mode='same')
    return matches


def filter_out_illegal_sequences(sequences, sim_item: LIgOSimulationItem, all_signals: list, max_signals_per_sequence: int):
    if max_signals_per_sequence > 1:
        raise NotImplementedError
    elif max_signals_per_sequence == -1:
        return sequences

    sim_signal_ids = [signal.id for signal in sim_item.signals]
    other_signals = [signal.id not in sim_signal_ids for signal in all_signals]
    signal_matrix = sequences.get_signal_matrix()
    legal_indices = np.logical_and(signal_matrix.sum(axis=1) <= max_signals_per_sequence,
                                   np.array(signal_matrix[:, other_signals] == 0).all(axis=1) if any(other_signals) else 1)

    return sequences[legal_indices]


def make_sequences_from_gen_model(sim_item: LIgOSimulationItem, sequence_batch_size: int, seed: int, sequence_path: Path, sequence_type: SequenceType,
                                  skew_model_for_signal: bool):
    sim_item.generative_model.generate_sequences(sequence_batch_size, seed=seed, path=sequence_path, sequence_type=sequence_type)

    if sim_item.generative_model.can_generate_from_skewed_gene_models() and skew_model_for_signal:
        v_genes = sorted(
            list(set(chain.from_iterable([[motif.v_call for motif in signal.motifs if motif.v_call] for signal in sim_item.signals]))))
        j_genes = sorted(
            list(set(chain.from_iterable([[motif.j_call for motif in signal.motifs if motif.j_call] for signal in sim_item.signals]))))

        sim_item.generative_model.generate_from_skewed_gene_models(v_genes=v_genes, j_genes=j_genes, seed=seed, path=sequence_path,
                                                                   sequence_type=sequence_type, batch_size=sequence_batch_size)


def make_bnp_annotated_sequences(sequences: GenModelAsTSV, all_signals: list, signal_matrix: np.ndarray, signal_positions: dict):
    kwargs = {**{s.id: signal_matrix[:, ind].astype(int) for ind, s in enumerate(all_signals)},
              **{f"{s.id}_positions": bnp.as_encoded_array(signal_positions[f"{s.id}_positions"], bnp.encodings.BaseEncoding) for ind, s in
                 enumerate(all_signals)},
              **{field_name: getattr(sequences, field_name) for field_name in GenModelAsTSV.__annotations__.keys()}}

    bnp_signal_matrix = make_signal_matrix_bnpdataclass(all_signals)(**kwargs)
    return bnp_signal_matrix


def make_signal_matrix_bnpdataclass(signals: list):
    signal_fields = [(s.id, int) for s in signals]
    signal_position_fields = [(f"{s.id}_positions", str) for s in signals]
    base_fields = [(field_name, field_type) for field_name, field_type in GenModelAsTSV.__annotations__.items()]

    functions = {"get_signal_matrix": lambda self: np.array([getattr(self, field) for field, t in signal_fields]).T,
                 "get_signal_names": lambda self: [field for field, t in signal_fields]}

    AnnotatedGenData = bnpdataclass(make_dataclass("AnnotatedGenData", fields=base_fields + signal_fields + signal_position_fields,
                                                   namespace=functions))

    return AnnotatedGenData


def build_imgt_positions(sequence_length: int, motif_instance: MotifInstance, sequence_region_type):
    assert sequence_length >= len(motif_instance), \
        "The motif instance is longer than sequence length. Remove the receptor_sequence from the repertoire or reduce max gap length " \
        "to be able to proceed."

    if sequence_region_type.to_string() == RegionType.IMGT_JUNCTION.name:
        return PositionHelper.gen_imgt_positions_from_junction_length(sequence_length)
    elif sequence_region_type.to_string() == RegionType.IMGT_CDR3.name:
        return PositionHelper.gen_imgt_positions_from_cdr3_length(sequence_length)
    else:
        raise NotImplementedError(f"IMGT positions here are defined only for CDR3 and JUNCTION region types, got {sequence_region_type}")


def choose_implant_position(imgt_positions, position_weights):
    imgt_implant_position = np.random.choice(list(position_weights.keys()), size=1, p=list(position_weights.values()))
    position = np.where(imgt_positions == imgt_implant_position)[0][0]
    return position


def check_iteration_progress(iteration: int, max_iterations: int):
    if iteration == round(max_iterations * 0.75):
        logging.warning(f"Iteration {iteration} out of {max_iterations} max iterations reached during rejection sampling.")
