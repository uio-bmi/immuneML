# quality: gold
import pickle
import shutil
import weakref
from uuid import uuid4

import numpy as np

from source.data_model.DatasetItem import DatasetItem
from source.data_model.cell.Cell import Cell
from source.data_model.cell.CellList import CellList
from source.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from source.data_model.receptor.ReceptorList import ReceptorList
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.receptor.receptor_sequence.SequenceAnnotation import SequenceAnnotation
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.simulation.implants.ImplantAnnotation import ImplantAnnotation
from source.util.NumpyHelper import NumpyHelper
from source.util.PathBuilder import PathBuilder


class Repertoire(DatasetItem):
    """
    Repertoire object consisting of sequence objects, each sequence attribute is stored as a list across all sequences and can be
    loaded separately. Internally, this class relies on numpy to store/load the data.
    """

    FIELDS = "sequence_aas,sequences,v_genes,j_genes,chains,counts,region_types,sequence_identifiers,cell_ids".split(",")

    @staticmethod
    def process_custom_lists(custom_lists):
        if custom_lists:
            field_list = list(custom_lists.keys())
            values = [custom_lists[field] for field in custom_lists.keys()]
            dtype = [(field, np.object) for field in custom_lists.keys()]
        else:
            field_list, values, dtype = [], [], []
        return field_list, values, dtype

    @classmethod
    def build(cls, sequence_aas: list, sequences: list, v_genes: list, j_genes: list, chains: list, counts: list, region_types: list,
              custom_lists: dict, sequence_identifiers: list, path: str, metadata=dict(), signals: dict = None, cell_ids: list = None):

        if sequence_identifiers is None or len(sequence_identifiers) == 0 or any(identifier is None for identifier in sequence_identifiers):
            sequence_identifiers = list(range(len(sequence_aas))) if sequence_aas is not None and len(sequence_aas) > 0 else list(range(len(sequences)))

        assert len(sequence_aas) == len(sequence_identifiers) or len(sequences) == len(sequence_identifiers)
        assert all(len(custom_lists[key]) == len(sequence_identifiers) for key in custom_lists) if custom_lists else True

        identifier = uuid4().hex

        data_filename = f"{path}{identifier}_data.npy"

        field_list, values, dtype = Repertoire.process_custom_lists(custom_lists)

        if signals:
            signals_filtered = {signal: signals[signal] for signal in signals if signal not in metadata["field_list"]}
            field_list_signals, values_signals, dtype_signals = Repertoire.process_custom_lists(signals_filtered)

            field_list.extend(field_list_signals)
            values.extend(values_signals)
            dtype.extend(dtype_signals)

        for field in Repertoire.FIELDS:
            if eval(field) is not None and not all(el is None for el in eval(field)):
                field_list.append(field)
                values.append(eval(field))
                dtype.append((field, np.object))

        dtype = np.dtype(dtype)

        repertoire_matrix = np.array(list(map(tuple, zip(*values))), order='F', dtype=dtype)
        np.save(data_filename, repertoire_matrix)

        metadata_filename = f"{path}{identifier}_metadata.pickle"
        metadata["field_list"] = field_list
        with open(metadata_filename, "wb") as file:
            pickle.dump(metadata, file)

        repertoire = Repertoire(data_filename, metadata_filename, identifier)
        return repertoire

    @classmethod
    def build_like(cls, repertoire, indices_to_keep: list, result_path: str):
        if indices_to_keep is not None and len(indices_to_keep) > 0:
            PathBuilder.build(result_path)

            data = repertoire.load_data()
            data = data[indices_to_keep]
            identifier = uuid4().hex

            data_filename = f"{result_path}{identifier}_data.npy"
            np.save(data_filename, data)

            metadata_filename = f"{result_path}{identifier}_metadata.pickle"
            shutil.copyfile(repertoire.metadata_filename, metadata_filename)

            new_repertoire = Repertoire(data_filename, metadata_filename, identifier)
            return new_repertoire
        else:
            return None

    @classmethod
    def build_from_sequence_objects(cls, sequence_objects: list, path: str, metadata: dict):

        assert all(isinstance(sequence, ReceptorSequence) for sequence in sequence_objects), \
            "Repertoire: all sequences have to be instances of ReceptorSequence class."

        sequence_aas, sequences, v_genes, j_genes, chains, counts, region_types, sequence_identifiers, cell_ids = [], [], [], [], [], [], [], [], []
        custom_lists = {key: [] for key in sequence_objects[0].metadata.custom_params} if sequence_objects[0].metadata else {}
        signals = {key: [] for key in metadata if "signal" in key}

        for sequence in sequence_objects:
            sequence_identifiers.append(sequence.identifier)
            sequence_aas.append(sequence.amino_acid_sequence)
            sequences.append(sequence.nucleotide_sequence)
            if sequence.metadata:
                v_genes.append(sequence.metadata.v_gene)
                j_genes.append(sequence.metadata.j_gene)
                chains.append(sequence.metadata.chain)
                counts.append(sequence.metadata.count)
                region_types.append(sequence.metadata.region_type)
                cell_ids.append(sequence.metadata.cell_id)
                for param in sequence.metadata.custom_params.keys():
                    custom_lists[param].append(sequence.metadata.custom_params[param] if param in sequence.metadata.custom_params else None)
            for key in signals.keys():
                if sequence.annotation and sequence.annotation.implants and len(sequence.annotation.implants) > 0:
                    signals[key].append(vars(sequence.annotation.implants[0]))
                else:
                    signals[key].append(None)

        return cls.build(sequence_aas, sequences, v_genes, j_genes, chains, counts, region_types, custom_lists,
                         sequence_identifiers, path, metadata, signals, cell_ids)

    def __init__(self, data_filename: str, metadata_filename: str, identifier: str):

        assert ".npy" in data_filename, \
            "Repertoire: the file representing the repertoire has to be in numpy binary format. Got {} instead."\
                .format(data_filename.rpartition(".")[1])

        self._data_filename = data_filename

        if metadata_filename:
            with open(metadata_filename, "rb") as file:
                self.metadata = pickle.load(file)
            self._fields = self.metadata["field_list"]

        self.metadata_filename = metadata_filename
        self.identifier = identifier
        self.data = None
        self.element_count = None

    def get_sequence_aas(self):
        return self.get_attribute("sequence_aas")

    def get_sequence_identifiers(self):
        return self.get_attribute("sequence_identifiers")

    def get_v_genes(self):
        return self.get_attribute("v_genes")

    def get_j_genes(self):
        return self.get_attribute("j_genes")

    def get_counts(self):
        return self.get_attribute("counts")

    def load_data(self):
        if self.data is None or (isinstance(self.data, weakref.ref) and self.data() is None):
            data = np.load(self._data_filename, allow_pickle=True)
            self.data = weakref.ref(data) if EnvironmentSettings.low_memory else data
        data = self.data() if EnvironmentSettings.low_memory else self.data
        self.element_count = data.shape[0]
        return data

    def get_attribute(self, attribute):
        data = self.load_data()
        if attribute in data.dtype.names:
            tmp = data[attribute]
            return tmp
        else:
            return None

    def get_attributes(self, attributes: list):
        data = self.load_data()
        result = {attribute: data[attribute] for attribute in attributes}
        return result

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

    def _make_sequence_object(self, row):

        fields = row.dtype.names

        keys = [key for key in row.dtype.names if "signal" in key]
        implants = []
        for key in keys:
            value_dict = row[key]
            if value_dict:
                implants.append(ImplantAnnotation(*value_dict))

        seq = ReceptorSequence(amino_acid_sequence=row["sequence_aas"] if "sequence_aas" in fields else None,
                               nucleotide_sequence=row["sequences"] if "sequences" in fields else None,
                               identifier=row["sequence_identifiers"] if "sequence_identifiers" in fields else None,
                               metadata=SequenceMetadata(v_gene=row["v_genes"] if "v_genes" in fields else None,
                                                         j_gene=row["j_genes"] if "j_genes" in fields else None,
                                                         chain=row["chains"] if "chains" in fields else None,
                                                         count=row["counts"] if "counts" in fields else None,
                                                         region_type=row["region_types"] if "region_types" in fields else None,
                                                         cell_id=row["cell_ids"] if "cell_ids" in fields else None,
                                                         custom_params={key: row[key] if key in fields else None
                                                                        for key in set(self._fields) - set(Repertoire.FIELDS)}),
                               annotation=SequenceAnnotation(implants=implants))

        return seq

    def _prepare_cell_lists(self):
        data = self.load_data()

        assert "cell_ids" in data.dtype.names and data["cell_ids"] is not None, \
            f"Repertoire: cannot return receptor objects in repertoire {self.identifier} since cell_ids are not specified. " \
            f"Existing fields are: {str(data.dtype.names)[1:-1]}"

        same_cell_lists = NumpyHelper.group_structured_array_by(data, "cell_ids")
        return same_cell_lists

    def _make_receptors(self, cell_content):
        sequences = ReceptorSequenceList()
        for item in cell_content:
            sequences.append(self._make_sequence_object(item))
        return ReceptorBuilder.build_objects(sequences)

    @property
    def sequences(self) -> ReceptorSequenceList:
        seqs = ReceptorSequenceList()

        data = self.load_data()

        for i in range(len(self.get_sequence_identifiers())):
            seq = self._make_sequence_object(data[i])
            seqs.append(seq)

        return seqs

    @property
    def receptors(self) -> ReceptorList:
        receptors = ReceptorList()

        same_cell_lists = self._prepare_cell_lists()

        for cell_content in same_cell_lists:
            receptors.extend(self._make_receptors(cell_content))

        return receptors

    @property
    def cells(self) -> CellList:
        cells = CellList()
        cell_lists = self._prepare_cell_lists()

        for cell_content in cell_lists:
            receptors = self._make_receptors(cell_content)
            cells.append(Cell(receptors))

        return cell_lists
