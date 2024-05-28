# quality: gold
import logging
import shutil
import weakref
from pathlib import Path
from typing import List, Any, Dict
from uuid import uuid4

import bionumpy as bnp
import numpy as np
import yaml

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.data_model.SequenceSet import SequenceSet
from immuneML.data_model.bnp_util import bnp_write_to_file, write_yaml, bnp_read_from_file, \
    make_dynamic_seq_set_dataclass, build_dynamic_bnp_dataclass_obj
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.PathBuilder import PathBuilder


class Repertoire(DatasetItem):
    """
    Repertoire object consisting of sequence objects, each sequence attribute is stored as a list across all sequences and can be
    loaded separately. Internally, this class relies on numpy to store/import_dataset the data.
    """

    FIELDS = ("sequence_aa", "sequence", "v_call", "j_call", "chain", "duplicate_count", "region_type", "frame_type",
              "sequence_id", "cell_id")

    @classmethod
    def build(cls, path: Path = None, metadata: dict = None, filename_base: str = None, identifier: str = None, **kwargs):

        identifier = uuid4().hex if identifier is None else identifier
        filename_base = filename_base if filename_base is not None else identifier
        data_filename = path / f"{filename_base}.tsv"

        bnp_object, type_dict = build_dynamic_bnp_dataclass_obj(kwargs)
        bnp_write_to_file(data_filename, bnp_object)

        metadata_filename = path / f"{filename_base}_metadata.yaml"
        metadata = {} if metadata is None else metadata
        metadata['type_dict'] = {key: SequenceSet.TYPE_TO_STR[val] for key, val in type_dict.items()}
        write_yaml(metadata_filename, metadata)

        repertoire = Repertoire(data_filename, metadata_filename, identifier)
        return repertoire

    @classmethod
    def build_like(cls, repertoire: 'Repertoire', indices_to_keep: list, result_path: Path, filename_base: str = None):
        PathBuilder.build(result_path)

        identifier = uuid4().hex
        filename_base = filename_base if filename_base is not None else identifier

        data_filename = result_path / f"{filename_base}.tsv"
        bnp_data = repertoire.load_bnp_data()
        bnp_data = bnp_data[indices_to_keep]

        bnp_write_to_file(data_filename, bnp_data)

        metadata_filename = result_path / f"{filename_base}_metadata.yaml"
        shutil.copyfile(repertoire.metadata_filename, metadata_filename)

        new_repertoire = Repertoire(data_filename, metadata_filename, identifier)
        return new_repertoire

    @classmethod
    def build_from_sequence_objects(cls, sequence_objects: list, path: Path, metadata: dict, filename_base: str = None,
                                    repertoire_id: str = None):

        assert all(isinstance(sequence, ReceptorSequence) for sequence in sequence_objects), \
            "Repertoire: all sequences have to be instances of ReceptorSequence class."

        sequence_aa, sequence, v_call, j_call, chain, duplicate_count, region_type, frame_type, sequence_id, cell_id = [], [], [], [], [], [], [], [], [], []
        custom_lists = {key: [] for key in sequence_objects[0].metadata.custom_params} if sequence_objects[
            0].metadata else {}
        signals = {}

        for index, seq in enumerate(sequence_objects):
            sequence_id.append(seq.sequence_id)
            sequence_aa.append(seq.sequence_aa)
            sequence.append(seq.sequence)
            if seq.metadata:
                v_call.append(seq.metadata.v_call)
                j_call.append(seq.metadata.j_call)
                chain.append(seq.metadata.chain.value if seq.metadata.chain else None)
                duplicate_count.append(seq.metadata.duplicate_count)
                region_type.append(seq.metadata.region_type.value if seq.metadata.region_type else None)
                frame_type.append(seq.metadata.frame_type.value if seq.metadata.frame_type else None)
                cell_id.append(seq.metadata.cell_id)
                for param in seq.metadata.custom_params.keys():
                    current_value = seq.metadata.custom_params[param] if param in seq.metadata.custom_params else None
                    if param in custom_lists:
                        custom_lists[param].append(current_value)
                    else:
                        custom_lists[param] = [None for _ in range(index)]
                        custom_lists[param].append(current_value)

        sequence_count = len(sequence)

        for signal in signals.keys():
            signal_info_count = len(signals[signal])
            if signal_info_count < sequence_count:
                signals[signal].extend([None for _ in range(sequence_count - signal_info_count)])

        return cls.build(sequence_aa=sequence_aa, sequence=sequence, v_call=v_call, j_call=j_call, chain=chain,
                         duplicate_count=duplicate_count, region_type=region_type, frame_type=frame_type,
                         sequence_id=sequence_id, path=path, metadata=metadata, cell_id=cell_id,
                         filename_base=filename_base, identifier=repertoire_id, **custom_lists, **signals)

    @property
    def _bnp_filename(self):
        return Path(str(self.data_filename) + ".tsv")

    def __init__(self, data_filename: Path, metadata_filename: Path, identifier: str):
        data_filename = Path(data_filename)
        metadata_filename = Path(metadata_filename) if metadata_filename is not None else None

        assert data_filename.suffix == ".tsv", \
            f"Repertoire: the file representing the repertoire has to be in tsv format. Got {data_filename.suffix} instead."

        self.data_filename = data_filename
        if metadata_filename:
            with metadata_filename.open("r") as file:
                self.metadata = yaml.safe_load(file)

        self.metadata_filename = metadata_filename
        self.identifier = identifier
        self.bnp_data = None
        self.element_count = None

    @property
    def _type_dict(self):
        return {key: SequenceSet.STR_TO_TYPE[val] for key, val in self.metadata["type_dict"].items()}

    @property
    def _buffer_type(self):
        return self._create_buffer_type_from_field_dict(self._type_dict)

    def get_sequence_aas(self, as_list: bool = False):
        return self.get_attribute("sequence_aa", as_list)

    def get_sequence_identifiers(self, as_list: bool = False):
        return self.get_attribute("sequence_id", as_list)

    def get_v_genes(self):
        return [v_call.split("*")[0] for v_call in self.get_attribute("v_call", as_list=True)]

    def get_j_genes(self):
        return [j_call.split("*")[0] for j_call in self.get_attribute("j_call", as_list=True)]

    def get_counts(self):
        counts = self.get_attribute("duplicate_count")
        if counts is not None:
            counts = np.array([int(count) if count != SequenceSet.get_neutral_value(int) else None for count in counts])
        return counts

    def get_chains(self):
        chains = self.get_attribute("chain")
        if chains is not None:
            chains = np.array(
                [Chain.get_chain(chain_str) if chain_str is not None else None for chain_str in chains.tolist()])
        return chains

    def get_attribute(self, attribute, as_list: bool = False):
        data = self.load_bnp_data()
        if attribute in self._type_dict:
            tmp = getattr(data, attribute)
            if as_list:
                return tmp.tolist()
            else:
                return tmp
        else:
            return None

    def get_attributes(self, attributes: list, as_list: bool = False):
        result = {}
        data = self.load_bnp_data()
        for attribute in attributes:
            if attribute in self._type_dict:
                result[attribute] = getattr(data, attribute).tolist() if as_list else getattr(data, attribute)
            else:
                logging.warning(
                    f"{Repertoire.__name__}: attribute {attribute} is not present in the repertoire {self.identifier}, skipping...")
        return result

    def get_region_type(self):
        region_types = self.get_attribute("region_type")
        if region_types is not None:
            region_types = set(region_types.tolist())
            assert len(region_types) == 1, f"Repertoire {self.identifier}: expected one region_type, found: {region_types}"
            return RegionType(region_types.pop())
        else:
            logging.warning(f'Repertoire {self.identifier}: region_types are not set for sequences.')
            return None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['bnp_data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.bnp_data = None

    def get_element_count(self):
        if self.element_count is None:
            self.load_bnp_data()
        return self.element_count

    def _make_sequence_object(self, row: dict):

        seq = ReceptorSequence(sequence_aa=row.get('sequence_aa', None),
                               sequence=row.get('sequence', None),
                               sequence_id=row.get('sequence_id', None),
                               metadata=SequenceMetadata(v_call=row.get('v_call', None),
                                                         j_call=row.get('j_call', None),
                                                         chain=row.get('chain', None),
                                                         duplicate_count=row.get('duplicate_count', None),
                                                         region_type=row.get('region_type', None),
                                                         frame_type=row.get('frame_type', None),
                                                         cell_id=row.get('cell_id', None),
                                                         custom_params={key: row[key] for key in row
                                                                        if key not in Repertoire.FIELDS}))

        return seq

    def _prepare_cell_lists(self):
        data = self.load_bnp_data()

        assert hasattr(data, 'cell_id') and getattr(data, 'cell_id') is not None, \
            f"Repertoire: cannot return receptor objects in repertoire {self.identifier} since cell_ids are not specified. " \
            f"Existing fields are: {str(dir(data))[1:-1]}"

        same_cell_lists = NumpyHelper.group_structured_array_by(data, "cell_id")
        return same_cell_lists

    def _make_receptors(self, cell_content):
        sequences = []
        for item in cell_content:
            sequences.append(self._make_sequence_object(item))
        return ReceptorBuilder.build_objects(sequences)

    def get_sequence_objects(self) -> List[ReceptorSequence]:
        """
        Lazily loads sequences from disk to reduce RAM consumption

        Returns:
            a list of ReceptorSequence objects
        """
        seqs = []

        data = self.load_bnp_data()

        for i in range(len(data)):
            seq = self._make_sequence_object(data.get_row_by_index(i))
            seqs.append(seq)

        del data
        self.bnp_data = None

        return seqs

    def load_bnp_data(self):
        if self.bnp_data is None or (isinstance(self.bnp_data, weakref.ref) and self.bnp_data() is None):
            data = bnp_read_from_file(self.data_filename, self._buffer_type)
            self.bnp_data = weakref.ref(data) if EnvironmentSettings.low_memory else data
        data = self.bnp_data() if EnvironmentSettings.low_memory else self.bnp_data
        self.element_count = len(data)
        return data

    @property
    def sequences(self):
        return self.get_sequence_objects()

    @property
    def receptors(self) -> List[Receptor]:
        """
        A property that creates a list of Receptor objects based on the cell_ids field in the following manner:
            - all sequences that have the same cell_id are grouped together
            - they are divided into groups based on the chain
            - all valid combinations of chains are created and used to make a receptor object - this means that if a cell has
              two beta (b1 and b2) and one alpha chain (a1), two receptor objects will be created: receptor1 (b1, a1), receptor2 (b2, a1)

        To avoid have multiple receptors in the same cell, use some of the preprocessing classes which could merge/eliminate multiple
        sequences. See the documentation of the preprocessing module for more information.

        Returns:
            List[Receptor]: a list of objects of Receptor class
        """
        data = self.load_bnp_data()
        receptors = []
        chains = data.chain.tolist()
        for i in range(0, len(data), 2):
            rows = data.get_rows_by_indices(i, i + 1)
            cls = ChainPair.get_chain_pair([Chain.get_chain(el) for el in chains[i:i+2]]).get_appropriate_receptor_class()
            receptors.append(cls.create_from_record(**rows))

        return receptors

    def _create_buffer_type_from_field_dict(self,
                                            type_dict: Dict[str, Any]) -> bnp.io.delimited_buffers.DelimitedBuffer:
        dataclass = make_dynamic_seq_set_dataclass(type_dict)
        return bnp.io.delimited_buffers.get_bufferclass_for_datatype(dataclass, delimiter='\t', has_header=True)
