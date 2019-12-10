# quality: gold
import pickle

import numpy as np

from source.data_model.DatasetItem import DatasetItem
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata


class SequenceRepertoireV3(DatasetItem):
    """
    Repertoire object consisting of sequence objects, each sequence attribute is stored as a list across all sequences and can be
    loaded separately. Internally, this class relies on h5py to store/load the data.
    """

    FIELDS = "sequence_aas,sequences,v_genes,j_genes,chains,counts,region_types,sequence_identifiers".split(",")
    FIELD_TYPES = {"sequence_aas": "<U30", "sequences": "<U100", "v_genes": "<U10", "j_genes": "<U10", "chains": "<U5", "counts": "int",
                   "region_types": "<U5", "sequence_identifiers": "<U10"}

    @classmethod
    def build(cls, sequence_aas: list, sequences: list, v_genes: list, j_genes: list, chains: list, counts: list, region_types: list,
                 custom_lists: dict, sequence_identifiers: list, path: str, metadata=None, identifier: str = None):

        assert len(sequence_aas) == len(sequence_identifiers) or len(sequences) == len(sequence_identifiers)
        assert all(len(custom_lists[key]) == len(sequence_identifiers) for key in custom_lists)

        data_filename = f"{path}{identifier}_data.npy"

        field_list = list(custom_lists.keys())
        values = [custom_lists[field] for field in custom_lists.keys()]
        dtype = [(field, np.object) for field in custom_lists.keys()]

        for field in SequenceRepertoireV3.FIELDS:
            if eval(field) and not all(el is None for el in eval(field)):
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

        repertoire = SequenceRepertoireV3(data_filename, metadata_filename, identifier)
        return repertoire

    @classmethod
    def build_from_sequence_objects(cls, sequence_objects: list, path: str, identifier: str, metadata: dict):

        assert all(isinstance(sequence, ReceptorSequence) for sequence in sequence_objects), \
            "SequenceRepertoire: all sequences have to be instances of ReceptorSequence class."

        sequence_aas, sequences, v_genes, j_genes, chains, counts, region_types, sequence_identifiers = [], [], [], [], [], [], [], []
        custom_lists = {key: [] for key in sequence_objects[0].metadata.custom_params} if sequence_objects[0].metadata else {}

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
                for param in custom_lists.keys():
                    custom_lists[param].append(sequence.metadata.custom_params[param] if param in sequence.metadata.custom_params else None)

        return cls.build(sequence_aas, sequences, v_genes, j_genes, chains, counts, region_types, custom_lists,
                         sequence_identifiers, path, metadata, identifier)

    def __init__(self, data_filename: str, metadata_filename, identifier: str):

        assert ".npy" in data_filename, \
            "SequenceRepertoire: the file representing the repertoire has to be in numpy binary format. Got {} instead."\
                .format(data_filename.rpartition(".")[1])

        self._data_filename = data_filename

        with open(metadata_filename, "rb") as file:
            self.metadata = pickle.load(file)

        self._fields = self.metadata["field_list"]
        self.metadata_filename = metadata_filename
        self.identifier = identifier
        self.data = None

    def get_attribute(self, name: str):
        raise NotImplementedError

    def _load(self):
        self.data = np.load(self._data_filename, allow_pickle=True)

    def get_sequence_aas(self):
        return self._get_attribute("sequence_aas")

    def get_sequence_identifiers(self):
        return self._get_attribute("sequence_identifiers")

    def get_v_genes(self):
        return self._get_attribute("v_genes")

    def get_j_genes(self):
        return self._get_attribute("j_genes")

    def _get_attribute(self, attribute):
        if self.data is None:
            self._load()
        if attribute in self.data.dtype.names:
            tmp = self.data[attribute]
            return tmp
        else:
            return None

    def free_memory(self):
        self.data = None

    @property
    def sequences(self):

        self._load()

        seqs = []

        for i in range(len(self.get_sequence_identifiers())):
            seq = ReceptorSequence(amino_acid_sequence=self.get_sequence_aas()[i] if self.get_sequence_aas() is not None else None,
                                   nucleotide_sequence=self._get_attribute("sequences")[i] if self._get_attribute("sequences") is not None else None,
                                   identifier=self.get_sequence_identifiers()[i] if self.get_sequence_identifiers() is not None else None,
                                   metadata=SequenceMetadata(v_gene=self.get_v_genes()[i] if self.get_v_genes() is not None else None,
                                                             j_gene=self.get_j_genes()[i] if self.get_j_genes() is not None else None,
                                                             chain=self._get_attribute("chains")[i] if self._get_attribute("chains") is not None else None,
                                                             count=self._get_attribute("counts")[i] if self._get_attribute("counts") is not None else None,
                                                             region_type=self._get_attribute("region_types")[i] if self._get_attribute("region_types") is not None else None,
                                                             custom_params={key: self._get_attribute(key)[i] if self._get_attribute(key) is not None else None
                                                                            for key in set(self._fields) - set(SequenceRepertoireV3.FIELDS)}))

            seqs.append(seq)

        return seqs
