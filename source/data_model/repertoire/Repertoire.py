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
    loaded separately. Internally, this class relies on numpy to store/import_dataset the data.
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

    @staticmethod
    def check_count(sequence_aas: list = None, sequences: list = None, custom_lists: dict = None) -> int:
        sequence_count = len(sequence_aas) if sequence_aas is not None else len(sequences) if sequences is not None else 0

        if sequences is not None and sequence_aas is not None:
            assert len(sequences) == len(sequence_aas), \
                f"Repertoire: there is a mismatch between number of nucleotide sequences ({len(sequences)}) and the number of amino acid " \
                f"sequences ({len(sequence_aas)})."

        assert all(len(custom_lists[key]) == sequence_count for key in custom_lists) if custom_lists else True, \
            f"Repertoire: there is a mismatch between the number of sequences and the number of attributes listed in " \
            f"{str(list(custom_lists.keys()))[1:-1]}"

        return sequence_count

    @classmethod
    def build(cls, sequence_aas: list = None, sequences: list = None, v_genes: list = None, j_genes: list = None, chains: list = None,
              counts: list = None, region_types: list = None, custom_lists: dict = None, sequence_identifiers: list = None,
              path: str = None, metadata=dict(), signals: dict = None, cell_ids: list = None):

        sequence_count = Repertoire.check_count(sequence_aas, sequences, custom_lists)

        if sequence_identifiers is None or len(sequence_identifiers) == 0 or any(identifier is None for identifier in sequence_identifiers):
            sequence_identifiers = list(range(sequence_count))

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

        self.data_filename = data_filename

        if metadata_filename:
            with open(metadata_filename, "rb") as file:
                self.metadata = pickle.load(file)
            self.fields = self.metadata["field_list"]

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
        counts = self.get_attribute("counts")
        if counts is not None:
            counts = counts.astype(float).astype(int)
        return counts

    def load_data(self):
        if self.data is None or (isinstance(self.data, weakref.ref) and self.data() is None):
            data = np.load(self.data_filename, allow_pickle=True)
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
                                                                        for key in set(self.fields) - set(Repertoire.FIELDS)}),
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
        receptors = ReceptorList()

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
