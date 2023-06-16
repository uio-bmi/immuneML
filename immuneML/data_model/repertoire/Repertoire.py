# quality: gold
import ast
import dataclasses
import logging
import shutil
import weakref
from dataclasses import make_dataclass
from pathlib import Path
from typing import List, Any, Dict
from uuid import uuid4

import bionumpy as bnp
import numpy as np
import yaml
from bionumpy import AminoAcidEncoding, DNAEncoding
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.data_model.DatasetItem import DatasetItem
from immuneML.data_model.cell.Cell import Cell
from immuneML.data_model.cell.CellList import CellList
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.implants.ImplantAnnotation import ImplantAnnotation
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.PathBuilder import PathBuilder


@bnpdataclass
class SequenceSet:
    sequence_aa: AminoAcidEncoding
    sequence: DNAEncoding
    v_call: str
    j_call: str
    region_type: str
    frame_type: str
    duplicate_count: int


STR_TO_TYPE = {'str': str, 'int': int, 'float': float, 'bool': bool, 'AminoAcidEncoding': bnp.encodings.AminoAcidEncoding}
TYPE_TO_STR = {val: key for key, val in STR_TO_TYPE.items()}


class Repertoire(DatasetItem):
    """
    Repertoire object consisting of sequence objects, each sequence attribute is stored as a list across all sequences and can be
    loaded separately. Internally, this class relies on numpy to store/import_dataset the data.
    """

    FIELDS = ("sequence_aa", "sequence", "v_call", "j_call", "chain", "duplicate_count", "region_type", "frame_type",
              "sequence_id", "cell_id")

    @staticmethod
    def process_custom_lists(custom_lists):
        try:
            if custom_lists:
                field_list = list(custom_lists.keys())
                values = [[NumpyHelper.get_numpy_representation(el) for el in custom_lists[field]] for field in
                          custom_lists.keys()]
                dtype = [(field, np.array(values[index]).dtype) for index, field in enumerate(custom_lists.keys())]
            else:
                field_list, values, dtype = [], [], []
            return field_list, values, dtype
        except Exception as e:
            print(f"Error occurred when processing custom lists to create a repertoire, custom lists: {custom_lists}.")
            raise e

    @staticmethod
    def check_count(sequence_aas: list = None, sequences: list = None, custom_lists: dict = None) -> int:
        sequence_count = len(sequence_aas) if sequence_aas is not None else len(
            sequences) if sequences is not None else 0

        if sequences is not None and sequence_aas is not None:
            assert len(sequences) == len(sequence_aas), \
                f"Repertoire: there is a mismatch between number of nucleotide sequences ({len(sequences)}) and the number of amino acid " \
                f"sequences ({len(sequence_aas)})."

        assert all(len(custom_lists[key]) == sequence_count for key in custom_lists) if custom_lists else True, \
            f"Repertoire: there is a mismatch between the number of sequences ({sequence_count}) and the number of attributes listed in " \
            f"{str(list(custom_lists.keys()))[1:-1]}"

        return sequence_count

    @classmethod
    def _build_bnpdataclass(cls, all_fields_dict: Dict[str, Any]):
        sequence_field_names = {field.name: field.type for field in dataclasses.fields(SequenceSet)}
        types = {}
        for key, value in all_fields_dict.items():
            if key in sequence_field_names:
                field_type = sequence_field_names[key]
            else:
                field_type = cls._get_field_type_from_values(value)
            types[key] = field_type
        dc = bnpdataclass(make_dataclass('DynamicSequenceSet', fields=types.items()))
        return dc(**all_fields_dict), types

    @classmethod
    def _get_field_type_from_values(cls, value):
        if isinstance(value, np.ndarray):
            return value.dtype
        if len(value) == 0:
            return str
        return type(value[0])

    @classmethod
    def build(cls, sequence_aa: list = None, sequence: list = None, v_call: list = None, j_call: list = None,
              chain: list = None, duplicate_count: list = None, region_type: list = None, frame_type: list = None,
              custom_lists: dict = None, sequence_id: list = None, path: Path = None, metadata: dict = None,
              signals: dict = None, cell_id: List[str] = None, filename_base: str = None, identifier: str = None):

        sequence_count = Repertoire.check_count(sequence_aa, sequence, custom_lists)

        if sequence_id is None or len(sequence_id) == 0 or any(identifier is None for identifier in sequence_id):
            sequence_id = np.arange(sequence_count).astype(str)

        identifier = uuid4().hex if identifier is None else identifier

        filename_base = filename_base if filename_base is not None else identifier

        data_filename = path / f"{filename_base}.npy"

        dtype, field_list, values = cls._create_field_specs(sequence_aa, sequence, v_call, j_call,
                                                            chain, duplicate_count, region_type, frame_type,
                                                            custom_lists, sequence_id, path, metadata,
                                                            signals, cell_id, filename_base, identifier)
        field_dict = dict(zip(field_list, values))
        bnp_object, type_dict = cls._build_bnpdataclass(field_dict)
        buffer_type = bnp.io.delimited_buffers.get_bufferclass_for_datatype(type(bnp_object), delimiter='\t',
                                                                            has_header=True)
        with bnp.open(str(data_filename) + '.tsv', 'w', buffer_type=buffer_type) as file:
            file.write(bnp_object)
        repertoire_matrix = np.array(list(map(tuple, zip(*values))), order='F', dtype=dtype)  # erase this
        np.save(str(data_filename), repertoire_matrix, allow_pickle=False)  # erase this

        metadata_filename = path / f"{filename_base}_metadata.yaml"
        metadata = {} if metadata is None else metadata
        # metadata["field_list"] = field_list
        metadata['type_dict'] = {key: TYPE_TO_STR[val] for key, val in type_dict.items()}
        with metadata_filename.open("w") as file:
            yaml.dump(metadata, file)

        repertoire = Repertoire(data_filename, metadata_filename, identifier, buffer_type=buffer_type)
        return repertoire

    @classmethod
    def _create_field_specs(cls, sequence_aa: list = None, sequence: list = None, v_call: list = None,
                            j_call: list = None,
                            chain: list = None, duplicate_count: list = None, region_type: list = None,
                            frame_type: list = None,
                            custom_lists: dict = None, sequence_id: list = None, path: Path = None,
                            metadata: dict = None,
                            signals: dict = None, cell_id: List[str] = None, filename_base: str = None,
                            identifier: str = None):
        field_list, values, dtype = Repertoire.process_custom_lists(custom_lists)
        if signals:
            signals_filtered = {f'{signal}_info': signals[signal] for signal in signals if
                                signal not in Repertoire.FIELDS}
            field_list_signals, values_signals, dtype_signals = Repertoire.process_custom_lists(signals_filtered)

            for index, field_name in enumerate(field_list_signals):
                if field_name not in field_list:
                    field_list.append(field_name)
                    values.append(values_signals[index])
                    dtype.append(dtype_signals[index])
        for field in Repertoire.FIELDS:
            if eval(field) is not None and not all(el is None for el in eval(field)):
                field_list.append(field)
                values.append(
                    [NumpyHelper.get_numpy_representation(val) if val is not None else np.nan for val in eval(field)])
                dtype.append((field, np.array(values[-1]).dtype))
        return dtype, field_list, values

    @classmethod
    def build_like(cls, repertoire: 'Repertoire', indices_to_keep: list, result_path: Path, filename_base: str = None):
        PathBuilder.build(result_path)

        data = repertoire.load_data()
        data = data[indices_to_keep]
        identifier = uuid4().hex
        filename_base = filename_base if filename_base is not None else identifier

        data_filename = result_path / f"{filename_base}.npy"
        np.save(str(data_filename), data)
        bnp_datafilename = data_filename.with_suffix('.npy.tsv')
        bnp_data= repertoire._load_bnp_data()
        bnp_data = bnp_data[indices_to_keep]
        with bnp.open(bnp_datafilename, 'w', buffer_type=repertoire._buffer_type) as f:
            f.write(bnp_data)
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
            sequence_id.append(seq.identifier)
            sequence_aa.append(seq.amino_acid_sequence)
            sequence.append(seq.nucleotide_sequence)
            if seq.metadata:
                v_call.append(seq.metadata.v_call)
                j_call.append(seq.metadata.j_call)
                chain.append(seq.metadata.chain)
                duplicate_count.append(seq.metadata.duplicate_count)
                region_type.append(seq.metadata.region_type)
                frame_type.append(seq.metadata.frame_type)
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
                         duplicate_count=duplicate_count,
                         region_type=region_type, frame_type=frame_type, custom_lists=custom_lists,
                         sequence_id=sequence_id, path=path,
                         metadata=metadata, signals=signals, cell_id=cell_id, filename_base=filename_base,
                         identifier=repertoire_id)

    @property
    def _bnp_filename(self):
        return Path(str(self.data_filename) + ".tsv")

    def __init__(self, data_filename: Path, metadata_filename: Path, identifier: str, buffer_type=None):
        data_filename = Path(data_filename)
        metadata_filename = Path(metadata_filename) if metadata_filename is not None else None

        self.__bnp_data = None
        assert data_filename.suffix == ".npy", \
            f"Repertoire: the file representing the repertoire has to be in numpy binary format. Got {data_filename.suffix} instead."

        self.data_filename = data_filename
        if metadata_filename:
            with metadata_filename.open("r") as file:
                self.metadata = yaml.safe_load(file)

        self.metadata_filename = metadata_filename
        self.identifier = identifier
        self.data = None
        self.element_count = None

    @property
    def _type_dict(self):
        return {key: STR_TO_TYPE[val] for key, val in self.metadata["type_dict"].items()}

    @property
    def _buffer_type(self):
        return self._create_buffer_type_from_field_dict(self._type_dict)

    def get_sequence_aas(self):
        return self.get_attribute("sequence_aa")

    def get_sequence_identifiers(self):
        return self.get_attribute("sequence_id")

    def get_v_genes(self):
        return self.get_attribute("v_call")

    def get_j_genes(self):
        return self.get_attribute("j_call")

    def get_counts(self):
        counts = self.get_attribute("duplicate_count")
        if counts is not None:
            counts = np.array([int(count) if not NumpyHelper.is_nan_or_empty(count) else None for count in counts])
        return counts

    def get_chains(self):
        chains = self.get_attribute("chain")
        if chains is not None:
            chains = np.array([Chain.get_chain(chain_str) if chain_str is not None else None for chain_str in chains])
        return chains

    def load_data(self):
        if self.data is None or (isinstance(self.data, weakref.ref) and self.data() is None):
            data = np.load(self.data_filename, allow_pickle=False)
            self.data = weakref.ref(data) if EnvironmentSettings.low_memory else data
        data = self.data() if EnvironmentSettings.low_memory else self.data
        self.element_count = data.shape[0]
        return data

    def get_attribute(self, attribute):
        data = self._load_bnp_data()
        if attribute in self._type_dict:
            tmp = getattr(data, attribute)
            return tmp
        else:
            return None

    def get_attributes(self, attributes: list):
        result = {}
        data = self._load_bnp_data()
        for attribute in attributes:
            if attribute in self._type_dict:
                result[attribute] = getattr(data, attribute)
            else:
                logging.warning(
                    f"{Repertoire.__name__}: attribute {attribute} is not present in the repertoire {self.identifier}, skipping...")
        return result

    def get_region_type(self):
        region_types = set(self.get_attribute("region_type").tolist())

        if 'nan' in region_types:
            region_types.remove('nan')

        assert len(region_types) == 1, f"Repertoire: expected one region_type, found: {region_types}"

        return RegionType(region_types.pop())

    def free_memory(self):
        self.data = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = None

    def get_element_count(self):
        if self.element_count is None:
            self.load_data()
        return self.element_count

    def _make_sequence_object(self, row, load_implants: bool = False):

        fields = row.dtype.names

        implants = []
        if load_implants:
            keys = [key for key in row.dtype.names if key not in Repertoire.FIELDS]
            for key in keys:
                value_dict = row[key]
                if value_dict:
                    try:
                        implants.append(ImplantAnnotation(**ast.literal_eval(value_dict)))
                    except (SyntaxError, ValueError, TypeError) as e:
                        implants.append(ImplantAnnotation(signal_id=key))

        seq = ReceptorSequence(amino_acid_sequence=row["sequence_aa"] if "sequence_aa" in fields else None,
                               nucleotide_sequence=row["sequence"] if "sequence" in fields else None,
                               identifier=row["sequence_id"] if "sequence_id" in fields else None,
                               metadata=SequenceMetadata(v_call=row["v_call"] if "v_call" in fields else None,
                                                         j_call=row["j_call"] if "j_call" in fields else None,
                                                         chain=row["chain"] if "chain" in fields else None,
                                                         duplicate_count=row[
                                                             "duplicate_count"] if "duplicate_count" in fields and not NumpyHelper.is_nan_or_empty(
                                                             row['duplicate_count']) else None,
                                                         region_type=row[
                                                             "region_type"] if "region_type" in fields else None,
                                                         frame_type=row[
                                                             "frame_type"] if "frame_type" in fields else "IN",
                                                         cell_id=row["cell_id"] if "cell_id" in fields else None,
                                                         custom_params={key: row[key] if key in fields else None
                                                                        for key in
                                                                        set(self._type_dict.keys()) - set(Repertoire.FIELDS)}))

        return seq

    def _prepare_cell_lists(self):
        data = self.load_data()

        assert "cell_id" in data.dtype.names and data["cell_id"] is not None, \
            f"Repertoire: cannot return receptor objects in repertoire {self.identifier} since cell_ids are not specified. " \
            f"Existing fields are: {str(data.dtype.names)[1:-1]}"

        same_cell_lists = NumpyHelper.group_structured_array_by(data, "cell_id")
        return same_cell_lists

    def _make_receptors(self, cell_content):
        sequences = []
        for item in cell_content:
            sequences.append(self._make_sequence_object(item))
        return ReceptorBuilder.build_objects(sequences)

    def get_sequence_objects(self, load_implants: bool = True) -> List[ReceptorSequence]:
        """
        Lazily loads sequences from disk to reduce RAM consumption

        Args:
            load_implants: whether implants should be parsed to objects and converted to ImplantAnnotations; if True, might slow down the loading

        Returns:
            a list of ReceptorSequence objects
        """
        seqs = []

        data = self.load_data()

        for i in range(len(self.get_sequence_identifiers())):
            seq = self._make_sequence_object(data[i], load_implants)
            seqs.append(seq)

        return seqs

    def _load_bnp_data(self):
        if self.__bnp_data is None or (isinstance(self.__bnp_data, weakref.ref) and self.__bnp_data() is None):
            data = self._read_bnp_data()
            self.__bnp_data = weakref.ref(data) if EnvironmentSettings.low_memory else data
        data = self.__bnp_data() if EnvironmentSettings.low_memory else self.__bnp_data
        self.element_count = len(data)
        return data

    @property
    def sequences(self):
        return self.get_sequence_objects(True)

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
            ReceptorList: a list of objects of Receptor class
        """
        receptors = []

        same_cell_lists = self._prepare_cell_lists()

        for cell_content in same_cell_lists:
            receptors.extend(self._make_receptors(cell_content))

        return receptors

    @property
    def cells(self) -> CellList:
        """
        A property that creates a list of Cell objects based on the cell_ids field in the following manner:
            - all sequences that have the same cell_id are grouped together
            - they are divided into groups based on the chain
            - all valid combinations of chains are created and used to make a receptor object - this means that if a cell has
              two beta (b1 and b2) and one alpha chain (a1), two receptor objects will be created: receptor1 (b1, a1), receptor2 (b2, a1)
            - an object of the Cell class is created from all receptors with the same cell_id created as described in the previous steps

        To avoid have multiple receptors in the same cell, use some of the preprocessing classes which could merge/eliminate multiple
        sequences. See the documentation of the preprocessing module for more information.

        Returns:
            CellList: a list of objects of Cell class
        """
        cells = CellList()
        cell_lists = self._prepare_cell_lists()

        for cell_content in cell_lists:
            receptors = self._make_receptors(cell_content)
            cells.append(Cell(receptors))

        return cell_lists

    def _read_bnp_data(self):
        return bnp.open(self._bnp_filename, buffer_type=self._buffer_type).read()

    def _create_buffer_type_from_field_dict(self, type_dict: Dict[str, Any])->bnp.io.delimited_buffers.DelimitedBuffer:
        dataclass = bnpdataclass(make_dataclass('DynamicSequenceSet', fields=type_dict.items()))
        return bnp.io.delimited_buffers.get_bufferclass_for_datatype(dataclass, delimiter='\t', has_header=True)

