import dataclasses
import logging
import uuid
from dataclasses import make_dataclass, fields as get_fields
from itertools import chain
from pathlib import Path
from typing import List, Dict, Union

import bionumpy as bnp
import numpy as np
from bionumpy import AminoAcidEncoding, DNAEncoding, EncodedRaggedArray, get_motif_scores
from bionumpy.bnpdataclass import bnpdataclass, BNPDataClass
from bionumpy.encodings import BaseEncoding
from bionumpy.io import delimited_buffers
from bionumpy.sequence.string_matcher import RegexMatcher, StringMatcher
from npstructures import RaggedArray

from immuneML import Constants
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.implants.MotifInstance import MotifInstance, MotifInstanceGroup
from immuneML.simulation.implants.PWM import PWM
from immuneML.simulation.implants.Signal import Signal, SignalPair
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


def get_signal_sequence_count(repertoire_count: int, signal_proportion: float, receptors_in_repertoire_count: int) -> int:
    return round(receptors_in_repertoire_count * signal_proportion) * repertoire_count


def get_sequence_per_signal_count(sim_item: SimConfigItem) -> dict:
    if sim_item.receptors_in_repertoire_count:
        sequence_count = sim_item.receptors_in_repertoire_count * sim_item.number_of_examples
        seq_with_signal_count = {signal.id: get_signal_sequence_count(sim_item.number_of_examples, proportion, sim_item.receptors_in_repertoire_count)
                                 for signal, proportion in sim_item.signal_proportions.items()}
        seq_without_signal_count = {'no_signal': sequence_count - sum(seq_with_signal_count.values())}

        return {**seq_with_signal_count, **seq_without_signal_count}
    elif len(sim_item.signals) == 1:
        return {sim_item.signals[0].id: sim_item.number_of_examples, 'no_signal': 0}
    elif len(sim_item.signals) == 0:
        return {'no_signal': sim_item.number_of_examples}
    else:
        raise NotImplementedError


def get_bnp_data(sequence_path, bnp_data_class):
    buff_type = delimited_buffers.get_bufferclass_for_datatype(bnp_data_class, delimiter='\t', has_header=True)

    with bnp.open(sequence_path, buffer_type=buff_type) as file:
        data = file.read()

    return data


def make_receptor_sequence_objects(sequences: BackgroundSequences, metadata, immune_events: dict, custom_params: list) -> List[ReceptorSequence]:
    return [ReceptorSequence(seq.sequence_aa.to_string(), seq.sequence.to_string(), identifier=uuid.uuid4().hex,
                             metadata=construct_sequence_metadata_object(seq, metadata, custom_params, immune_events)) for seq in sequences]


def construct_sequence_metadata_object(sequence, metadata: dict, custom_params, immune_events: dict) -> SequenceMetadata:
    custom = {}

    for key, key_type in custom_params:
        if 'position' in key:
            custom[key] = getattr(sequence, key).to_string()
        else:
            custom[key] = getattr(sequence, key).item()

    return SequenceMetadata(custom_params={**metadata, **custom, **immune_events},
                            v_call=sequence.v_call.to_string(), j_call=sequence.j_call.to_string(), region_type=sequence.region_type.to_string())


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


def make_sequence_paths(path: Path, signals: List[Signal]) -> Dict[str, Path]:
    tmp_path = PathBuilder.build(path / 'processed_sequences')
    sequence_paths = {signal.id: tmp_path / f'{signal.id}.tsv' for signal in signals}
    sequence_paths['no_signal'] = tmp_path / 'no_signal.tsv'
    return sequence_paths


def get_allowed_positions(signal: Signal, sequence_array: RaggedArray, region_type: RegionType):
    sequence_lengths = sequence_array.lengths
    if bool(signal.sequence_position_weights):
        signal_positions = [key for key, val in signal.sequence_position_weights.items() if val > 0]
        allowed_positions = RaggedArray([[pos in signal_positions for pos in PositionHelper.gen_imgt_positions_from_length(seq_len, region_type)]
                                         for seq_len in sequence_lengths])
    else:
        allowed_positions = None

    return allowed_positions


def get_region_type(sequences) -> RegionType:
    if hasattr(sequences, "region_type") and \
        np.all([el.to_string() == getattr(sequences, 'region_type')[0].to_string() for el in getattr(sequences, 'region_type')]):
        return RegionType[getattr(sequences, 'region_type')[0].to_string()]
    else:
        raise RuntimeError(f"The region types could not be obtained.")


def annotate_sequences(sequences, is_amino_acid: bool, all_signals: list, annotated_dc):
    encoding = AminoAcidEncoding if is_amino_acid else DNAEncoding
    sequence_array = sequences.sequence_aa if is_amino_acid else sequences.sequence

    signal_matrix = np.zeros((len(sequence_array), len(all_signals)))
    signal_positions = {}

    for index, signal in enumerate(all_signals):
        signal_pos_col = None
        allowed_positions = get_allowed_positions(signal, sequence_array, get_region_type(sequences))
        matches_gene = match_genes(signal.v_call, sequences.v_call, signal.j_call, sequences.j_call)

        for motifs in signal.get_all_motif_instances(SequenceType.AMINO_ACID if is_amino_acid else SequenceType.NUCLEOTIDE):
            matches = None

            if isinstance(motifs, MotifInstanceGroup):
                matches = match_motif_group(motifs, encoding, sequence_array, matches_gene, matches)
            else:
                matches = match_motif_regexes(motifs, encoding, sequence_array, matches_gene, matches)

            if allowed_positions is not None:
                matches = np.logical_and(matches, allowed_positions)

            signal_pos_col = np.logical_or(signal_pos_col, matches) if signal_pos_col is not None else matches
            signal_matrix[:, index] = np.logical_or(signal_matrix[:, index], np.logical_or.reduce(matches, axis=1))
            # TODO: we want to remove a receptor if it has multiple matches of the motif -> xor? or somehow removed?

        np_mask = RaggedArray(np.where(signal_pos_col.ravel(), "1", "0"), shape=signal_pos_col.shape)
        signal_positions[f'{signal.id}_positions'] = ['m' + "".join(np_mask[ind, :]) for ind in range(len(signal_pos_col))]

    signal_matrix = make_bnp_annotated_sequences(sequences, annotated_dc, all_signals, signal_matrix, signal_positions)

    logging.info(f"Annotated {len(sequences)} sequences with signal information.")

    return signal_matrix


def match_motif_regexes(motifs, encoding, sequence_array, matches_gene, matches):
    for motif in motifs:

        matches_motif = match_motif(motif, encoding, sequence_array)
        if matches is None:
            matches = np.logical_and(matches_motif, matches_gene)
        else:
            matches = np.logical_or(matches, np.logical_and(matches_motif, matches_gene))

    return matches


def match_motif_group(motif_group: list, encoding, sequence_array, matches_gene, matches):
    """Match if two motifs co-occur in the same sequence"""

    assert len(motif_group) == 2, len(motif_group)

    matches_motif_1 = match_motif_regexes(motif_group[0], encoding, sequence_array, matches_gene, None)
    matches_motif_2 = match_motif_regexes(motif_group[1], encoding, sequence_array, matches_gene, None)
    matches_motif = np.zeros_like(matches_motif_1)
    selection = np.logical_and(matches_motif_1.any(axis=1), matches_motif_2.any(axis=1))
    matches_motif[selection] = np.logical_or(matches_motif_1[selection], matches_motif_2[selection])

    return np.logical_or(matches_motif, matches) if matches is not None else matches_motif


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


def match_motif(motif: Union[str, PWM], encoding, sequence_array):
    if isinstance(motif, str):
        matcher = RegexMatcher(motif, encoding=encoding)
        matches = matcher.rolling_window(sequence_array, mode='same')
    else:
        matches = get_motif_scores(sequence_array, motif.pwm_matrix) > motif.threshold
    return matches


def filter_out_illegal_sequences(sequences, sim_item: SimConfigItem, all_signals: list, max_signals_per_sequence: int):
    if max_signals_per_sequence > 2:
        raise NotImplementedError
    elif max_signals_per_sequence == -1 or all_signals is None or len(all_signals) == 0:
        return sequences

    sim_signal_ids = list(set(chain.from_iterable([signal.id.split(Constants.SIGNAL_DELIMITER) for signal in sim_item.signals])))
    other_signals = [signal.id not in sim_signal_ids for signal in all_signals]
    signal_matrix = sequences.get_signal_matrix()
    legal_indices = np.logical_and(signal_matrix.sum(axis=1) <= max_signals_per_sequence,
                                   np.array(signal_matrix[:, other_signals] == 0).all(axis=1) if any(other_signals) else 1)

    return sequences[legal_indices]


def make_signal_metadata(sim_item, signals) -> Dict[str, bool]:
    metadata = {}
    for signal in sim_item.signals:
        if isinstance(signal, SignalPair):
            metadata[signal.signal1.id] = True if not sim_item.is_noise else False
            metadata[signal.signal2.id] = True if not sim_item.is_noise else False
        else:
            metadata[signal.id] = True if not sim_item.is_noise else False

    return {**metadata, **{signal.id: False for signal in signals if signal.id not in metadata}}


def make_repertoire_from_sequences(sequences: BNPDataClass, repertoires_path, sim_item: SimConfigItem, signals: List[Signal], custom_fields: list) \
    -> Repertoire:
    metadata = {**make_signal_metadata(sim_item, signals), **sim_item.immune_events, 'sim_item': sim_item.name}
    rep_data = prepare_data_for_repertoire_obj(sequences, custom_fields)
    return Repertoire.build(**rep_data, path=repertoires_path, metadata=metadata)


def make_bnp_annotated_sequences(sequences: BackgroundSequences, bnp_data_class, all_signals: list, signal_matrix: np.ndarray,
                                 signal_positions: dict):
    kwargs = {**{s.id: signal_matrix[:, ind].astype(int) for ind, s in enumerate(all_signals)},
              **{f"{s.id}_positions": bnp.as_encoded_array(signal_positions[f"{s.id}_positions"], bnp.encodings.BaseEncoding) for ind, s in
                 enumerate(all_signals)},
              **{field_name: getattr(sequences, field_name) for field_name in BackgroundSequences.__annotations__.keys()}}

    dc_fields = get_fields(bnp_data_class)
    if any([f'observed_{s.id}' in dc_fields for s in all_signals]):
        kwargs = {**kwargs,
                  **{f'observed_{s.id}': signal_matrix[:, ind].astype(int) for ind, s in enumerate(all_signals)}}

    return bnp_data_class(**kwargs)


def make_annotated_dataclass(annotation_fields: list, signals: list):
    functions = {
        "get_signal_matrix": lambda self: np.array([getattr(self, signal.id) for signal in signals]).T if signals and len(signals) > 0 else None,
        "get_signal_names": lambda self: [signal.id for signal in signals]}

    return bnpdataclass(make_dataclass("AnnotatedGenData", namespace=functions, bases=tuple([BackgroundSequences]), fields=annotation_fields))


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


def check_sequence_count(sim_item, sequences: BackgroundSequences):
    assert len(sequences) == sim_item.receptors_in_repertoire_count, \
        f"Error when simulating repertoire, needed {sim_item.receptors_in_repertoire_count} sequences, " \
        f"but got {len(sequences)}."


def prepare_data_for_repertoire_obj(sequences: BNPDataClass, custom_fields: list) -> dict:
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    shuffled_sequences = sequences[indices]

    custom_lists = {}
    for field, field_type in custom_fields:
        if field_type is int or field_type is float:
            custom_lists[field] = getattr(shuffled_sequences, field)
        else:
            custom_lists[field] = [el.to_string() for el in getattr(shuffled_sequences, field)]

    default_lists = {}
    for field in dataclasses.fields(shuffled_sequences):
        if field.name not in custom_lists:
            if isinstance(getattr(shuffled_sequences, field.name), EncodedRaggedArray):
                default_lists[field.name] = [el.to_string() for el in getattr(shuffled_sequences, field.name)]
            else:
                default_lists[field.name] = getattr(shuffled_sequences, field.name)

    return {**{"custom_lists": custom_lists}, **default_lists}


def update_seqs_without_signal(max_count, annotated_sequences, seqs_no_signal_path: Path):
    if max_count > 0:
        signal_matrix = annotated_sequences.get_signal_matrix()
        selection = signal_matrix.sum(axis=1) == 0 if signal_matrix is not None else np.ones(len(annotated_sequences), dtype=bool)
        data_to_write = annotated_sequences[selection][:max_count]
        if len(data_to_write) > 0:
            write_bnp_data(data=data_to_write, path=seqs_no_signal_path)
        return max_count - len(data_to_write)
    else:
        return max_count


def update_seqs_with_signal(max_counts: dict, annotated_sequences, all_signals, sim_item_signals, seqs_with_signal_path: dict):
    all_signal_ids = [signal.id for signal in all_signals]
    signal_matrix = annotated_sequences.get_signal_matrix()

    for signal in sim_item_signals:
        if max_counts[signal.id] > 0:
            if isinstance(signal, SignalPair):
                selection = signal_matrix[:, [all_signal_ids.index(sid) for sid in signal.id.split(Constants.SIGNAL_DELIMITER)]].astype(bool).all(axis=1)
            else:
                selection = signal_matrix[:, all_signal_ids.index(signal.id)].astype(bool)
                selection = np.logical_and(selection, signal_matrix.sum(axis=1) == 1)
            data_to_write = annotated_sequences[selection][:max_counts[signal.id]]
            if len(data_to_write) > 0:
                write_bnp_data(data=data_to_write, path=seqs_with_signal_path[signal.id])
            max_counts[signal.id] -= len(data_to_write)

    return max_counts


def get_signal_sequences(sequences, bnp_data_class, used_seq_count: dict, sim_item: SimConfigItem, sequence_paths: Dict[str, Path]):
    for signal in sim_item.signals:

        skip_rows = used_seq_count[signal.id]
        n_rows = round(sim_item.receptors_in_repertoire_count * sim_item.signal_proportions[signal])

        sequences_sig = get_bnp_data(sequence_paths[signal.id], bnp_data_class)[skip_rows:skip_rows + n_rows]

        used_seq_count[signal.id] += n_rows

        if sequences is None:
            sequences = sequences_sig
        else:
            sequences = merge_dataclass_objects([sequences, sequences_sig])

    return sequences, used_seq_count


def get_no_signal_sequences(used_seq_count: dict, seqs_no_signal_count: int, bnp_data_class, sequence_paths: Dict[str, Path]):
    if sequence_paths['no_signal'].is_file() and seqs_no_signal_count > 0:
        skip_rows = used_seq_count['no_signal']
        used_seq_count['no_signal'] = used_seq_count['no_signal'] + seqs_no_signal_count
        return get_bnp_data(sequence_paths['no_signal'], bnp_data_class)[skip_rows:skip_rows + seqs_no_signal_count], used_seq_count
    else:
        return None, used_seq_count


def needs_seqs_with_signal(sequence_per_signal_count: dict) -> bool:
    return (sum(sequence_per_signal_count.values()) - sequence_per_signal_count['no_signal']) > 0