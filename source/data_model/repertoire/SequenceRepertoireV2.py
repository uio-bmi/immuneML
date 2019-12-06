# quality: gold

import uuid

from source.data_model.DatasetItem import DatasetItem
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata


class SequenceRepertoireV2(DatasetItem):

    @classmethod
    def build(cls, sequence_aas: list, sequences: list, v_genes: list, j_genes: list, chains: list, counts: list, region_types: list,
                 extra_lists: dict, sequence_identifiers: list, metadata=None, identifier: str = None):

        REQUIRED_FIELDS = "sequence_as,sequences".split(",")
        assert all(rf in kwArgs for rf in REQUIRED_FIELDS)

        for attributes in [sequence_aas, sequences, v_genes, j_genes, chains, counts, region_types]:
            assert len(attributes) == len(sequence_identifiers) or len(attributes) == 0

        for key in extra_lists:
            assert len(extra_lists[key]) == len(sequence_identifiers) or len(extra_lists[key]) == 0

        FIELDS = "sequence_aas,sequences".split(",")
        for field in FIELDS:
            self._std_lists[field] = eval(field)
        self._lists["sequence_identifiers"] = sequence_identifiers
        self._lists["sequence_aas"] = sequence_aas
        self._sequences = sequences
        self._v_genes = v_genes
        self._j_genes = j_genes
        self._chains = chains
        self._counts = counts
        self._region_types = region_types
        self._extra_lists = extra_lists
        self.metadata = metadata
        self.identifier = identifier if identifier is not None else str(uuid.uuid1())

    def __init__(self, filename: str, metadata: dict):


    def get_identifier(self):
        return self.identifier

    def get_attribute(self, name: str):
        raise NotImplementedError

    def _store(self):
        pass

    def _load(cls):
        pass

r = AdaptiveLoader.createRep(fn)
#r = Rep(fn="rep1.numpy", meta={})
...
r.getAAs()
#r._load()
..
r.flush()
...
r.getAAs()
#r._load

    @property
    def sequences(self):
        for i in range(len(self._sequence_aas)):
            yield ReceptorSequence(amino_acid_sequence=self._sequence_aas[i], nucleotide_sequence=self._sequences[i],
                                   identifier=self._sequence_identifiers[i],
                                   metadata=SequenceMetadata(v_gene=self._v_genes[i], j_gene=self._j_genes[i], chain=self._chains[i],
                                                             count=self._counts[i], region_type=self._region_types[i],
                                                             custom_params={key: self._extra_lists[key][i] for key in self._extra_lists}))
