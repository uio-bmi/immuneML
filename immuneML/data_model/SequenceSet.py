from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from bionumpy import AminoAcidEncoding, DNAEncoding, get_bufferclass_for_datatype

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import ChainPair, Chain, RegionType
from immuneML.data_model.bnp_util import get_field_type_from_values, \
    extend_dataclass_with_dynamic_fields, bnp_write_to_file, write_yaml, bnp_read_from_file, read_yaml


@dataclass
class ReceptorSequence:
    sequence_id: str = None
    sequence: DNAEncoding = None
    sequence_aa: AminoAcidEncoding = None
    productive: bool = None
    vj_in_frame: bool = None
    stop_codon: bool = None
    locus: str = None
    locus_species: str = None
    v_call: str = None
    d_call: str = None
    j_call: str = None
    c_call: str = None
    metadata: dict = field(default_factory=dict)
    duplicate_count: int = None
    cell_id: str = None


@dataclass
class Receptor:
    chain_pair: str
    chain_1: ReceptorSequence
    chain_2: ReceptorSequence
    receptor_id: str
    cell_id: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        new_metadata = {}
        for key, val in self.metadata.items():
            if isinstance(val, list) and len(val) == 1:
                new_metadata[key] = val[0]
            else:
                new_metadata[key] = val
        self.metadata = new_metadata


@dataclass
class Repertoire:
    data_filename: Path = None
    metadata_filename: Path = None
    metadata: dict = None
    identifier: str = None
    dynamic_fields: list = None

    bnp_dataclass = None
    element_count: int = None
    _buffer_type = None

    @classmethod
    def build(cls, path: Path, metadata: dict, filename_base: str = None, identifier: str = None, **kwargs):
        identifier = uuid4().hex if identifier is None else identifier
        filename_base = filename_base if filename_base is not None else identifier
        data_filename = path / f"{filename_base}.tsv"

        bnp_dc, type_dict = build_dynamic_airr_sequence_set_dataclass(kwargs)
        data = bnp_dc(**kwargs)
        bnp_write_to_file(data_filename, data)

        metadata_filename = path / f"{filename_base}_metadata.yaml"
        metadata = {} if metadata is None else metadata
        metadata['type_dict_dynamic_fields'] = {key: AIRRSequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()}
        write_yaml(metadata_filename, metadata)

        repertoire = Repertoire(data_filename, metadata_filename, metadata, identifier, list(type_dict.keys()),
                                bnp_dataclass=bnp_dc, element_count=len(data))
        return repertoire

    def __post_init__(self):
        if not self.metadata:
            self.metadata = read_yaml(self.metadata_filename)
        if not self.dynamic_fields:
            self.dynamic_fields = list(self.metadata['type_dict_dynamic_fields'].items())
        if not self.bnp_dataclass:
            self.bnp_dataclass = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, self.dynamic_fields)

    @property
    def buffer_type(self):
        if not self._buffer_type:
            self._buffer_type = get_bufferclass_for_datatype(self.bnp_dataclass, delimiter='\t', has_header=True)
        return self._buffer_type

    @property
    def data(self) -> AIRRSequenceSet:
        return bnp_read_from_file(self.data_filename, self.buffer_type, self.bnp_dataclass)

    def sequences(self, region_type: RegionType) -> List[ReceptorSequence]:
        return make_sequences_from_data(self.data, self.dynamic_fields, region_type)

    @property
    def receptors(self) -> List[Receptor]:
        return make_receptors_from_data(self.data, self.dynamic_fields, f'Repertoire {self.identifier}')


def build_dynamic_airr_sequence_set_dataclass(all_fields_dict: Dict[str, Any]):
    sequence_field_names = {seq_field.name: seq_field.type for seq_field in fields(AIRRSequenceSet)}
    types = {}

    for key, values in all_fields_dict.items():
        if key not in sequence_field_names:
            field_type = get_field_type_from_values(values)
            types[key] = field_type

    if types:
        dc = extend_dataclass_with_dynamic_fields(AIRRSequenceSet, tuple(types.items()))
    else:
        dc = AIRRSequenceSet
    return dc, types


def make_sequences_from_data(data, dynamic_fields: list, region_type):
    seqs = []
    for el in data.to_iter():
        seq, seq_aa = get_sequence_value(el, region_type)
        seqs.append(ReceptorSequence(el.sequence_id, seq, seq_aa, el.productive, el.vj_in_frame, el.stop_codon,
                                     el.locus, el.locus_species, el.v_call, el.d_call, el.j_call, el.c_call,
                                     {dynamic_field: getattr(el, dynamic_field) for dynamic_field in dynamic_fields}))
    return seqs


def make_receptors_from_data(data: AIRRSequenceSet, dynamic_fields: list, location):
    df = data.topandas()
    receptors = []
    for name, group in df.groupby('cell_id'):
        assert group.shape[0] == 2, \
            (f"{location}: receptor objects cannot be created from the data file, there are "
             f"{group.shape[0]} sequences with cell id {group['cell_id'].unique()[0]}.")
        sorted_group = group.sort_values(by='locus')
        seqs = [ReceptorSequence(el.sequence_id, el.sequence, el.sequence_aa, el.productive,
                                 el.vj_in_frame, el.stop_codon, el.locus, el.locus_species, el.v_call, el.d_call,
                                 el.j_call, el.c_call, {dynamic_field: getattr(el, dynamic_field) for dynamic_field
                                                        in Repertoire.dynamic_fields()}) for index, el in
                sorted_group.iterrows()]

        receptor = Receptor(chain_pair=ChainPair.get_chain_pair([Chain.get_chain(locus) for locus in group.locus]),
                            chain_1=seqs[0], chain_2=seqs[1], cell_id=group['cell_id'].unique()[0],
                            receptor_id=uuid4().hex,
                            metadata={key: list({seqs[0].metadata[key], seqs[1].metadata[key]})
                                      for key in dynamic_fields})
        receptors.append(receptor)

    return receptors


def get_sequence_value(el: AIRRSequenceSet, region_type):
    if region_type.FULL_SEQUENCE:
        return el.sequence, el.sequence_aa
    else:
        return getattr(el, region_type.value), getattr(el, region_type.value + "_aa")
