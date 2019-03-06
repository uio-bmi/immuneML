import os
import pickle
from unittest import TestCase

from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata


class TestSequenceMatcher(TestCase):

    def test_match(self):
        repertoire = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAAAAA", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
                                           ReceptorSequence(amino_acid_sequence="CCCCCC", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
                                           ReceptorSequence(amino_acid_sequence="AAAACC", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
                                           ReceptorSequence(amino_acid_sequence="TADQVF", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J3"))],
                                metadata=RepertoireMetadata(sample=Sample("CD123"), custom_params={"CD": True}))

        with open("./rep0.pkl", "wb") as file:
            pickle.dump(repertoire, file)

        dataset = Dataset(filenames=["./rep0.pkl"])
        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J3"))]

        matcher = SequenceMatcher()
        result = matcher.match(dataset, sequences, 2)

        self.assertTrue("repertoires" in result)
        self.assertEqual(1, len(result["repertoires"][0]["sequences"][3]["matching_sequences"]))
        self.assertTrue(result["repertoires"][0]["metadata"]["CD"])
        self.assertEqual(1, len(result["repertoires"]))

        os.remove("./rep0.pkl")

    def test_match_repertoire(self):
        repertoire = Repertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAAAAA", metadata=SequenceMetadata(chain="A")),
                                           ReceptorSequence(amino_acid_sequence="CCCCCC", metadata=SequenceMetadata(chain="A")),
                                           ReceptorSequence(amino_acid_sequence="AAAACC", metadata=SequenceMetadata(chain="A")),
                                           ReceptorSequence(amino_acid_sequence="TADQVF", metadata=SequenceMetadata(chain="A"))],
                                metadata=RepertoireMetadata(sample=Sample("CD123"), custom_params={"CD": True}))

        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(chain="A")),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(chain="A"))]

        matcher = SequenceMatcher()
        result = matcher.match_repertoire(repertoire, 0, sequences, 2)

        self.assertTrue("sequences" in result)
        self.assertTrue("repertoire" in result)
        self.assertTrue("repertoire_index" in result)

        self.assertEqual(4, len(result["sequences"]))
        self.assertEqual(1, len(result["sequences"][0]["matching_sequences"]))
        self.assertEqual(0, len(result["sequences"][1]["matching_sequences"]))
        self.assertEqual(1, len(result["sequences"][2]["matching_sequences"]))
        self.assertEqual(1, len(result["sequences"][3]["matching_sequences"]))

        self.assertEqual(3, len([r for r in result["sequences"] if len(r["matching_sequences"]) > 0]))
        self.assertTrue(result["metadata"]["CD"])

