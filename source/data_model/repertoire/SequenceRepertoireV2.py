# quality: gold

import uuid

import h5py
import numpy as np

from source.data_model.DatasetItem import DatasetItem
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.util.Util import determine_list_datatype


class SequenceRepertoireV2(DatasetItem):
    """
    Repertoire object consisting of sequence objects, each sequence attribute is stored as a list across all sequences and can be
    loaded separately. Internally, this class relies on h5py to store/load the data.
    """

    FIELDS = "sequence_aas,sequences,v_genes,j_genes,chains,counts,region_types,sequence_identifiers".split(",")
    STRING_FORMAT = h5py.string_dtype()

    @classmethod
    def build(cls, sequence_aas: list, sequences: list, v_genes: list, j_genes: list, chains: list, counts: list, region_types: list,
                 custom_lists: dict, sequence_identifiers: list, path: str, metadata=None, identifier: str = None):

        assert len(sequence_aas) == len(sequence_identifiers) or len(sequences) == len(sequence_identifiers)
        assert all(len(custom_lists[key]) == len(sequence_identifiers) for key in custom_lists)

        filename = '{}{}.hdf5'.format(path, identifier)

        with h5py.File(filename, 'a') as f:

            repertoire_group = f.create_group("repertoire")
            repertoire_group.attrs.update(metadata)
            repertoire_group.attrs["identifier"] = identifier if identifier is not None else str(uuid.uuid1())
            repertoire_group.attrs["sequence_count"] = len(sequence_identifiers)

            for field in cls.FIELDS:
                field_value = eval(field)
                if field_value is not None and not all(el is None for el in field_value):
                    dt = determine_list_datatype(field_value, SequenceRepertoireV2.STRING_FORMAT)
                    repertoire_group.create_dataset(field, data=np.array(field_value, dtype=dt), dtype=dt)

            for field in custom_lists:
                dt = determine_list_datatype(custom_lists[field], SequenceRepertoireV2.STRING_FORMAT)
                repertoire_group.create_dataset(field, data=np.array(custom_lists[field], dtype=dt))

        repertoire = SequenceRepertoireV2(filename, metadata, identifier)
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

    def __init__(self, filename: str, metadata: dict, identifier: str):

        assert ".hdf5" in filename, \
            "SequenceRepertoire: the file representing the repertoire has to be in HDF5 format. Got {} instead."\
                .format(filename.rpartition(".")[1])

        self._filename = filename
        self.metadata = metadata
        self.identifier = identifier
        self.data = {**{"custom_lists": {}}, **{key: None for key in SequenceRepertoireV2.FIELDS}}
        for field in SequenceRepertoireV2.FIELDS:
            self.data[field] = None

    def get_attribute(self, name: str):
        raise NotImplementedError

    def _load(self, fields=None):
        if fields is None:
            fields = SequenceRepertoireV2.FIELDS
        elif not isinstance(fields, list):
            fields = [fields]

        with h5py.File(self._filename, 'r') as f:

            group = f["repertoire"]

            for field in fields:
                if field in group.keys():
                    tmp = np.empty((group.attrs["sequence_count"],), dtype=group[field].dtype)
                    group[field].read_direct(tmp)
                    tmp[tmp == ''] = None
                    if field in SequenceRepertoireV2.FIELDS:
                        self.data[field] = tmp
                    else:
                        self.data["custom_lists"][field] = tmp

    def get_sequence_aas(self):
        return self._get_attribute("sequence_aas")

    def get_sequence_identifiers(self):
        return self._get_attribute("sequence_identifiers")

    def get_v_genes(self):
        return self._get_attribute("v_genes")

    def get_j_genes(self):
        return self._get_attribute("j_genes")

    def get_custom_attribute(self, attribute):
        if attribute not in self.data["custom_lists"]:
            self._load(attribute)
        return self.data["custom_lists"][attribute]

    def _get_attribute(self, attribute):
        if attribute not in self.data or self.data[attribute] is None:
            self._load(attribute)
        return self.data[attribute]

    def free_memory(self):
        self.data = {**{"custom_lists": {}}, **{key: None for key in SequenceRepertoireV2.FIELDS}}

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
                                                             custom_params={key: self.get_custom_attribute(key)[i] if self.get_custom_attribute(key) is not None else None
                                                                            for key in self.data["custom_lists"].keys()}))

            seqs.append(seq)

        return seqs
