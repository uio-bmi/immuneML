import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import dill
from bionumpy import DNAEncoding, get_bufferclass_for_datatype, AminoAcidEncoding

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import ChainPair, Chain, RegionType
from immuneML.data_model.bnp_util import get_field_type_from_values, \
    extend_dataclass_with_dynamic_fields, bnp_write_to_file, write_yaml, bnp_read_from_file, read_yaml, \
    build_dynamic_bnp_dataclass_obj
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


@dataclass
class ReceptorSequence:
    sequence_id: str = ''
    sequence: DNAEncoding = ''
    sequence_aa: AminoAcidEncoding = ''
    productive: str = 'T'
    vj_in_frame: str = 'T'
    stop_codon: str = 'F'
    locus: str = ''
    locus_species: str = ''
    v_call: str = ''
    d_call: str = ''
    j_call: str = ''
    c_call: str = ''
    metadata: dict = field(default_factory=dict)
    duplicate_count: int = -1
    cell_id: str = ''

    @property
    def v_gene(self):
        if self.v_call and len(self.v_call) > 0:
            return self.v_call.split("*")[0]
        else:
            return ""

    def get_sequence(self, sequence_type: SequenceType = SequenceType.AMINO_ACID):
        return self.sequence_aa if sequence_type == SequenceType.AMINO_ACID else self.sequence

    def get_attribute(self, attr_name):
        try:
            if hasattr(self, attr_name):
                return getattr(self, attr_name)
            else:
                return self.metadata.get(attr_name)
        except KeyError as e:
            logging.error(f"ReceptorSequence object has no attribute {e}. In metadata, "
                          f"it has: {list(self.metadata.keys()) if isinstance(self.metadata, dict) else []}")
            raise e

    def __post_init__(self):
        if self.sequence_id is None or self.sequence_id == "":
            self.sequence_id = uuid4().hex


@dataclass
class Receptor:
    chain_pair: ChainPair
    chain_1: ReceptorSequence
    chain_2: ReceptorSequence
    receptor_id: str = None
    cell_id: str = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        new_metadata = {}
        for key, val in self.metadata.items():
            if isinstance(val, list) and len(val) == 1:
                new_metadata[key] = val[0]
            else:
                new_metadata[key] = val
        self.metadata = new_metadata

        if self.receptor_id is None:
            self.receptor_id = uuid4().hex

        if isinstance(self.chain_pair, str):
            self.chain_pair = ChainPair[self.chain_pair]

        setattr(self, Chain.get_chain(self.chain_pair.value[0]).name.lower(), self.chain_1)
        setattr(self, Chain.get_chain(self.chain_pair.value[1]).name.lower(), self.chain_2)


@dataclass
class Repertoire:
    data_filename: Path = None
    metadata_filename: Path = None
    metadata: dict = None
    identifier: str = None
    dynamic_fields: dict = None
    _element_count: int = None
    _bnp_dataclass: bytes = None
    _buffer_type: bytes = None

    def __post_init__(self):
        if not self.identifier:
            self.identifier = uuid4().hex
        if not self.metadata:
            self.metadata = read_yaml(self.metadata_filename)
        if not self.dynamic_fields and 'type_dict_dynamic_fields' in self.metadata:
            self.dynamic_fields = self.metadata.get('type_dict_dynamic_fields', {})

    @property
    def element_count(self):
        if self._element_count is None:
            self._element_count = len(self.data)
        return self._element_count

    @classmethod
    def build_from_dc_object(cls, path: Path, metadata: dict, filename_base: str = None, identifier: str = None,
                             data=None, type_dict: dict = None):
        identifier = uuid4().hex if identifier is None else identifier
        filename_base = filename_base if filename_base is not None else identifier
        data_filename = path / f"{filename_base}.tsv"

        bnp_write_to_file(data_filename, data)

        metadata_filename = path / f"{filename_base}_metadata.yaml"
        metadata = {} if metadata is None else metadata

        if not type_dict:
            type_dict = {f.name: f.type for f in fields(data)
                         if f.name not in [airr_field.name for airr_field in fields(AIRRSequenceSet)]}

        metadata['type_dict_dynamic_fields'] = {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()}
        write_yaml(metadata_filename, metadata)

        repertoire = Repertoire(data_filename, metadata_filename, metadata, identifier, type_dict, len(data))
        return repertoire

    @classmethod
    def build(cls, path: Path, metadata: dict, filename_base: str = None, identifier: str = None, **kwargs):

        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(kwargs)
        el_count = list(kwargs.values())[0]
        el_count = len(el_count) if isinstance(el_count, list) else el_count.shape[0]
        kwargs_with_missing_fields = {
            **kwargs,
            **{k: [AIRRSequenceSet.get_neutral_value(t) for _ in range(el_count)]
               for k, t in AIRRSequenceSet.get_field_type_dict().items() if k not in kwargs}
        }
        data = bnp_dc(**kwargs_with_missing_fields)

        return Repertoire.build_from_dc_object(path, metadata, filename_base, identifier, data, type_dict)

    @classmethod
    def build_like(cls, repertoire: 'Repertoire', indices_to_keep, result_path: Path, filename_base: str):
        identifier = uuid4().hex
        filename_base = filename_base if filename_base is not None else identifier
        data = bnp_read_from_file(repertoire.data_filename, repertoire.buffer_type)
        data = data[indices_to_keep]

        PathBuilder.build(result_path)

        data_filename = result_path / f"{filename_base}.tsv"
        bnp_write_to_file(data_filename, data)

        metadata_filename = result_path / f"{filename_base}_metadata.yaml"
        metadata = read_yaml(repertoire.metadata_filename)
        write_yaml(metadata_filename, metadata)

        repertoire = Repertoire(data_filename, metadata_filename, metadata, identifier,
                                dynamic_fields=repertoire.dynamic_fields)
        repertoire._element_count = len(data)
        return repertoire

    @classmethod
    def build_from_sequences(cls, sequences: List[ReceptorSequence], result_path: Path, filename_base: str = None,
                             metadata: dict = None, region_type: RegionType = RegionType.IMGT_CDR3):
        identifier = uuid4().hex
        filename_base = filename_base if filename_base is not None else identifier

        data = make_airr_seq_set_object_from_sequences(sequences, region_type)

        data_filename = result_path / f"{filename_base}.tsv"
        bnp_write_to_file(data_filename, data)

        dynamic_fields = {f.name: f.type for f in fields(data)
                          if f not in [airr_f.name for airr_f in fields(AIRRSequenceSet)]}

        metadata_filename = result_path / f"{filename_base}_metadata.yaml"
        metadata = {} if metadata is None else metadata
        metadata['type_dict_dynamic_fields'] = dynamic_fields
        write_yaml(metadata_filename, metadata)

        repertoire = Repertoire(data_filename, metadata_filename, metadata, identifier,
                                _bnp_dataclass=type(data), dynamic_fields=dynamic_fields)
        repertoire._element_count = len(data)

        return repertoire

    @property
    def bnp_dataclass(self):
        if not self._bnp_dataclass:
            if self.dynamic_fields:
                bnp_dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, list(self.dynamic_fields.items()))
                self._bnp_dataclass = dill.dumps(bnp_dc)
                return bnp_dc
            else:
                self._bnp_dataclass = dill.dumps(AIRRSequenceSet)
                return AIRRSequenceSet
        elif isinstance(self._bnp_dataclass, bytes):
            return dill.loads(self._bnp_dataclass)
        else:
            return self._bnp_dataclass

    @property
    def buffer_type(self):
        if not self._buffer_type:
            buffer_type = get_bufferclass_for_datatype(self.bnp_dataclass, delimiter='\t', has_header=True)
            self._buffer_type = dill.dumps(buffer_type)
            return buffer_type
        elif isinstance(self._buffer_type, bytes):
            return dill.loads(self._buffer_type)
        else:
            return self._buffer_type

    @property
    def data(self) -> AIRRSequenceSet:
        return bnp_read_from_file(self.data_filename, self.buffer_type, self.bnp_dataclass)

    def sequences(self, region_type: RegionType = RegionType.IMGT_CDR3) -> List[ReceptorSequence]:
        return make_sequences_from_data(self.data, self.dynamic_fields, region_type)

    def receptors(self, region_type: RegionType) -> List[Receptor]:
        return make_receptors_from_data(self.data, self.dynamic_fields, f'Repertoire {self.identifier}',
                                        region_type)

    def get_element_count(self):
        if not self.element_count:
            self.element_count = len(self.data)
        return self.element_count


def build_dynamic_airr_sequence_set_dataclass(all_fields_dict: Dict[str, Any]):
    sequence_field_names = {seq_field.name: seq_field.type for seq_field in fields(AIRRSequenceSet)}
    types = {}

    for key, values in all_fields_dict.items():
        if key not in sequence_field_names:
            field_type = get_field_type_from_values(values)
            types[key] = field_type

    if types:
        dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, list(types.items()))
    else:
        dc = AIRRSequenceSet
    return dc, types


def make_sequences_from_data(data, dynamic_fields: dict, region_type: RegionType = RegionType.IMGT_CDR3):
    seqs = []
    for el in data.to_iter():
        seq, seq_aa = get_sequence_value(el, region_type)
        seqs.append(ReceptorSequence(sequence_id=el.sequence_id, sequence=seq, sequence_aa=seq_aa,
                                     productive=el.productive, vj_in_frame=el.vj_in_frame, stop_codon=el.stop_codon,
                                     locus=el.locus, v_call=el.v_call,
                                     d_call=el.d_call if hasattr(el, 'd_call') else '',
                                     c_call=getattr(el, 'c_call', ''),
                                     j_call=el.j_call, duplicate_count=el.duplicate_count,
                                     metadata={dynamic_field: getattr(el, dynamic_field)
                                               for dynamic_field in dynamic_fields.keys()}))
    return seqs


def make_receptors_from_data(data: AIRRSequenceSet, dynamic_fields: dict, location,
                             region_type: RegionType = RegionType.IMGT_CDR3):
    df = data.topandas()
    receptors = []
    for name, group in df.groupby('cell_id'):
        assert group.shape[0] == 2, \
            (f"{location}: receptor objects cannot be created from the data file, there are "
             f"{group.shape[0]} sequences with cell id {group['cell_id'].unique()[0]}.")
        sorted_group = group.sort_values(by='locus')
        seqs = [ReceptorSequence(sequence_id=el.sequence_id, sequence=getattr(el, region_type.value),
                                 sequence_aa=getattr(el, region_type.value + "_aa"),
                                 productive=el.productive, vj_in_frame=el.vj_in_frame, stop_codon=el.stop_codon,
                                 locus=el.locus, locus_species=el.locus_species, v_call=el.v_call,
                                 d_call=el.d_call, j_call=el.j_call, c_call=el.c_call,
                                 duplicate_count=el.duplicate_count,
                                 metadata={dynamic_field: getattr(el, dynamic_field)
                                           for dynamic_field in dynamic_fields.keys()})
                for index, el in sorted_group.iterrows()]

        if 'receptor_id' in group and group['receptor_id'].nunique() == 1:
            receptor_id = group['receptor_id'].unique()[0]
        else:
            receptor_id = uuid4().hex

        receptor = Receptor(chain_pair=ChainPair.get_chain_pair([Chain.get_chain(locus) for locus in group.locus]),
                            chain_1=seqs[0], chain_2=seqs[1], cell_id=group['cell_id'].unique()[0],
                            receptor_id=receptor_id,
                            metadata={key: list({seqs[0].metadata[key], seqs[1].metadata[key]})
                                      for key in dynamic_fields.keys()})
        receptors.append(receptor)

    return receptors


def make_airr_seq_set_object_from_sequences(sequences: List[ReceptorSequence],
                                            region_type: RegionType = RegionType.IMGT_CDR3):
    seq_fields = {key: [] for key in vars(ReceptorSequence()).keys()
                  if key not in ['metadata', 'sequence', 'sequence_aa']}
    seq_content_fields = {region_type.value: [], region_type.value + "_aa": []}

    dynamic_fields = {}
    for index, sequence in enumerate(sequences):
        seq_content_fields[region_type.value].append(sequence.sequence)
        seq_content_fields[region_type.value + "_aa"].append(sequence.sequence_aa)

        for key in seq_fields.keys():
            seq_fields[key].append(getattr(sequence, key))
        if sequence.metadata:
            for key in sequence.metadata.keys():
                if key in dynamic_fields:
                    dynamic_fields[key].append(sequence.metadata[key])
                elif index == 0:
                    dynamic_fields[key] = [sequence.metadata[key]]
                else:
                    raise RuntimeError("Sequences cannot be converted to AIRRSequenceSet because the metadata "
                                       "attributes between sequences differ.")

    all_fields_dict = {**seq_fields, **dynamic_fields, **seq_content_fields}
    return build_dynamic_bnp_dataclass_obj(all_fields_dict)[0]


def get_sequence_value(el: AIRRSequenceSet, region_type: RegionType = RegionType.IMGT_CDR3):
    if region_type == RegionType.FULL_SEQUENCE:
        return el.sequence, el.sequence_aa
    else:
        return getattr(el, region_type.value, ''), getattr(el, region_type.value + "_aa", '')
