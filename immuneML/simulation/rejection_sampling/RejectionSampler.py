import dataclasses
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bionumpy import EncodedRaggedArray
from bionumpy.bnpdataclass import BNPDataClass

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.simulation.util.util import get_signal_sequence_count, get_sequence_per_signal_count, make_sequences_from_gen_model, get_bnp_data, \
    annotate_sequences, filter_out_illegal_sequences, write_bnp_data, check_iteration_progress
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class RejectionSampler:
    sim_item: LIgOSimulationItem
    sequence_type: SequenceType
    all_signals: List[Signal]
    sequence_batch_size: int
    max_iterations: int
    seqs_no_signal_path: Path = None
    seqs_with_signal_path: dict = None
    seed: int = 1
    export_pgens: bool = None

    MAX_SIGNALS_PER_SEQUENCE = 1
    MAX_MOTIF_POSITION_LENGTH = 10

    @property
    def is_amino_acid(self):
        return self.sequence_type == SequenceType.AMINO_ACID

    @property
    def fields(self) -> list:
        return [(field.name, field.type) for field in dataclasses.fields(GenModelAsTSV)] + \
               [(signal.id, int) for signal in self.all_signals] + \
               [(f"{signal.id}_positions", str) for signal in self.all_signals]

    @property
    def use_p_gens(self) -> bool:
        return self.export_pgens and self.sim_item.generative_model.can_compute_p_gens()

    def make_repertoires(self, path: Path) -> List[Repertoire]:
        repertoires_path = PathBuilder.build(path / "repertoires")

        sequence_per_signal_count = get_sequence_per_signal_count(self.sim_item)

        self._make_background_sequences(path / f"tmp_{self.sim_item.name}", sequence_per_signal_count)
        repertoires = []
        used_seq_count = {**{'no_signal': 0}, **{signal.id: 0 for signal in self.sim_item.signals}}

        for i in range(self.sim_item.number_of_examples):
            seqs_no_signal_count = self.sim_item.receptors_in_repertoire_count - get_signal_sequence_count(repertoire_count=1,
                                                                                                           sim_item=self.sim_item) * len(
                self.sim_item.signals)

            custom_columns = self._get_custom_keys(with_p_gens=False)

            sequences, used_seq_count = self._get_no_signal_sequences(used_seq_count=used_seq_count, seqs_no_signal_count=seqs_no_signal_count, columns=custom_columns)
            sequences, used_seq_count = self._add_signal_sequences(sequences, custom_columns, used_seq_count)
            sequences = self._add_pgens(sequences)

            self._check_sequence_count(sequences)

            repertoire = self._make_repertoire_from_sequences(sequences, repertoires_path)

            repertoires.append(repertoire)

        shutil.rmtree(path / "tmp", ignore_errors=True)

        return repertoires

    def _get_custom_keys(self, with_p_gens: bool):
        keys = [(sig.id, int) for sig in self.all_signals] + [(f'{signal.id}_positions', str) for signal in self.all_signals]
        if with_p_gens:
            keys += [('p_gen', float)]
        return keys

    def _make_repertoire_from_sequences(self, sequences: BNPDataClass, repertoires_path) -> Repertoire:
        metadata = {**self._make_signal_metadata(), **self.sim_item.immune_events}
        rep_data = self._prepare_data_for_repertoire_obj(sequences)
        return Repertoire.build(**rep_data, path=repertoires_path, metadata=metadata)

    def _prepare_data_for_repertoire_obj(self, sequences: BNPDataClass) -> dict:
        custom_keys = self._get_custom_keys(with_p_gens=self.use_p_gens)

        custom_lists = {}
        for field, field_type in custom_keys:
            if field_type is int or field_type is float:
                custom_lists[field] = getattr(sequences, field)
            else:
                custom_lists[field] = [el.to_string() for el in getattr(sequences, field)]

        default_lists = {}
        for field in dataclasses.fields(sequences):
            if field.name not in custom_lists:
                if isinstance(getattr(sequences, field.name), EncodedRaggedArray):
                    default_lists[field.name] = [el.to_string() for el in getattr(sequences, field.name)]
                else:
                    default_lists[field.name] = getattr(sequences, field.name)

        return {**{"custom_lists": custom_lists}, **default_lists}

    def _make_signal_metadata(self) -> dict:
        return {**{signal.id: True if not self.sim_item.is_noise else False for signal in self.sim_item.signals},
                **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

    def _get_no_signal_sequences(self, used_seq_count: dict, seqs_no_signal_count: int, columns):
        if self.seqs_no_signal_path.is_file() and seqs_no_signal_count > 0:
            skip_rows = used_seq_count['no_signal']
            used_seq_count['no_signal'] = used_seq_count['no_signal'] + seqs_no_signal_count
            return get_bnp_data(self.seqs_no_signal_path, columns)[skip_rows:skip_rows + seqs_no_signal_count], used_seq_count
        else:
            return None, used_seq_count

    def _add_signal_sequences(self, sequences, columns, used_seq_count: dict):

        for signal in self.sim_item.signals:

            skip_rows = used_seq_count[signal.id]
            n_rows = round(self.sim_item.receptors_in_repertoire_count * self.sim_item.repertoire_implanting_rate)

            sequences_sig = get_bnp_data(self.seqs_with_signal_path[signal.id], columns)[skip_rows:skip_rows + n_rows]

            used_seq_count[signal.id] += n_rows

            if sequences is None:
                sequences = sequences_sig
            else:
                sequences = merge_dataclass_objects([sequences, sequences_sig])

        return sequences, used_seq_count

    def _check_sequence_count(self, sequences: BNPDataClass):
        assert len(sequences) == self.sim_item.receptors_in_repertoire_count, \
            f"{RejectionSampler.__name__}: error when simulating repertoire, needed {self.sim_item.receptors_in_repertoire_count} sequences, " \
            f"but got {len(sequences)}."

    def _make_background_sequences(self, path: Path, sequence_per_signal_count: dict):
        background_path = PathBuilder.build(path / f"tmp_{self.sim_item.name}")
        self._setup_tmp_sequence_paths(background_path)
        iteration = 1

        while (sum(sequence_per_signal_count.values()) != 0) and iteration <= self.max_iterations:
            sequence_path = PathBuilder.build(background_path / f"gen_model/") / f"tmp_{self.seed}.tsv"

            needs_seqs_with_signal = self._needs_seqs_with_signal(sequence_per_signal_count)
            make_sequences_from_gen_model(self.sim_item, self.sequence_batch_size, self.seed, sequence_path, self.sequence_type,
                                          needs_seqs_with_signal)
            self.seed += 1

            background_sequences = get_bnp_data(sequence_path)
            annotated_sequences = annotate_sequences(background_sequences, self.is_amino_acid, self.all_signals)
            annotated_sequences = filter_out_illegal_sequences(annotated_sequences, self.sim_item, self.all_signals,
                                                               RejectionSampler.MAX_SIGNALS_PER_SEQUENCE)

            sequence_per_signal_count['no_signal'] = self._update_seqs_without_signal(sequence_per_signal_count['no_signal'], annotated_sequences)
            sequence_per_signal_count = self._update_seqs_with_signal(sequence_per_signal_count, annotated_sequences)

            check_iteration_progress(iteration, self.max_iterations)
            iteration += 1

        if iteration == self.max_iterations and sum(sequence_per_signal_count.values()) != 0:
            raise RuntimeError(f"{RejectionSampler.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self)}.\n")

    def _needs_seqs_with_signal(self, sequence_per_signal_count: dict) -> bool:
        return (sum(sequence_per_signal_count.values()) - sequence_per_signal_count['no_signal']) > 0

    def _setup_tmp_sequence_paths(self, path):
        if self.seqs_with_signal_path is None:
            self.seqs_with_signal_path = {signal.id: path / f"sequences_with_signal_{signal.id}.tsv" for signal in self.sim_item.signals}
        if self.seqs_no_signal_path is None:
            self.seqs_no_signal_path = path / "sequences_no_signal.tsv"

    def _update_seqs_without_signal(self, max_count, annotated_sequences):
        if max_count > 0:
            selection = annotated_sequences.get_signal_matrix().sum(axis=1) == 0
            data_to_write = annotated_sequences[selection][:max_count]
            if len(data_to_write) > 0:
                write_bnp_data(data=data_to_write, path=self.seqs_no_signal_path)
            return max_count - len(data_to_write)
        else:
            return max_count

    def _update_seqs_with_signal(self, max_counts: dict, annotated_sequences):

        all_signal_ids = [signal.id for signal in self.all_signals]
        signal_matrix = annotated_sequences.get_signal_matrix()

        for signal in self.sim_item.signals:
            if max_counts[signal.id] > 0:
                selection = signal_matrix[:, all_signal_ids.index(signal.id)].astype(bool)
                data_to_write = annotated_sequences[selection][:max_counts[signal.id]]
                if len(data_to_write) > 0:
                    write_bnp_data(data=data_to_write, path=self.seqs_with_signal_path[signal.id])
                max_counts[signal.id] -= len(data_to_write)

        return max_counts

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path) -> List[ReceptorSequence]:

        assert len(self.sim_item.signals) in [0, 1], f"RejectionSampler: for sequence datasets, only 0 or 1 signal per sequence are supported, " \
                                                     f"but {len(self.sim_item.signals)} were specified."

        PathBuilder.build(path)
        sequences = None

        seqs_per_signal_count = get_sequence_per_signal_count(self.sim_item)
        self._make_background_sequences(path, seqs_per_signal_count)

        if len(self.sim_item.signals) == 0:
            sequences = get_bnp_data(self.seqs_no_signal_path, self.fields)
        else:
            for signal in self.sim_item.signals:
                signal_sequences = get_bnp_data(self.seqs_with_signal_path[signal.id], self.fields)
                if sequences is not None:
                    sequences = merge_dataclass_objects([sequences, signal_sequences])
                else:
                    sequences = signal_sequences

        sequences = self._add_pgens(sequences)
        sequences = self._make_receptor_sequence_objects(sequences)

        self._remove_tmp_paths(path)

        return sequences

    def _make_sequence_metadata(self):
        return {**{signal.id: False if self.sim_item.is_noise else True for signal in self.sim_item.signals},
                **{signal.id: False for signal in self.all_signals if signal not in self.sim_item.signals}}

    def _add_pgens(self, sequences: BNPDataClass):
        if not hasattr(sequences, 'p_gen') and self.export_pgens and self.sim_item.generative_model.can_compute_p_gens():
            p_gens = self.sim_item.generative_model.compute_p_gens(sequences, self.sequence_type)
            sequences = sequences.add_fields({'p_gen': p_gens}, {'p_gen': float})

        return sequences

    def _make_receptor_sequence_objects(self, sequences: BNPDataClass) -> List[ReceptorSequence]:
        custom_params = self._get_custom_keys(self.use_p_gens)
        metadata = self._make_sequence_metadata()

        return [ReceptorSequence(seq.sequence_aa.to_string(), seq.sequence.to_string(), identifier=uuid.uuid4().hex,
                                 metadata=self._construct_sequence_metadata_object(seq, metadata, custom_params)) for seq in sequences]

    def _construct_sequence_metadata_object(self, sequence, metadata: dict, custom_params) -> SequenceMetadata:
        custom = {}

        for key, key_type in custom_params:
            if 'position' in key:
                custom[key] = getattr(sequence, key).to_string()
            else:
                custom[key] = getattr(sequence, key).item()

        return SequenceMetadata(
            custom_params={**metadata, **custom, **self.sim_item.immune_events},
            v_call=sequence.v_call.to_string(), j_call=sequence.j_call.to_string(), region_type=sequence.region_type.to_string())

    def _remove_tmp_paths(self, path: Path):
        if (path / f"tmp_{self.sim_item.name}").exists():
            shutil.rmtree(path / f"tmp_{self.sim_item.name}")
