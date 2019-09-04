import pickle
import shutil
from unittest import TestCase

from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceMatcher(TestCase):

    def test_match(self):
        repertoire = SequenceRepertoire(sequences=[
            ReceptorSequence(amino_acid_sequence="AAAAAA", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
            ReceptorSequence(amino_acid_sequence="CCCCCC", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
            ReceptorSequence(amino_acid_sequence="AAAACC", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
            ReceptorSequence(amino_acid_sequence="TADQVF", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J3"))],
                                metadata=RepertoireMetadata(sample=Sample("CD123"), custom_params={"CD": True}))

        path = EnvironmentSettings.root_path + "test/tmp/seqmatch/"
        PathBuilder.build(path)
        with open(path + "rep0.pkl", "wb") as file:
            pickle.dump(repertoire, file)

        dataset = RepertoireDataset(filenames=[path + "rep0.pkl"])
        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J2")),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(chain="A", v_gene="V1", j_gene="J3"))]

        matcher = SequenceMatcher()
        result = matcher.match(dataset, sequences, 2, SequenceMatchingSummaryType.PERCENTAGE)

        self.assertTrue("repertoires" in result)
        self.assertEqual(1, len(result["repertoires"][0]["sequences"][3]["matching_sequences"]))
        self.assertTrue(result["repertoires"][0]["metadata"]["CD"])
        self.assertEqual(1, len(result["repertoires"]))

        shutil.rmtree(path)

    def test_match_repertoire(self):
        repertoire = SequenceRepertoire(sequences=[ReceptorSequence(amino_acid_sequence="AAAAAA", metadata=SequenceMetadata(chain="A", count=3)),
                                                   ReceptorSequence(amino_acid_sequence="CCCCCC", metadata=SequenceMetadata(chain="A", count=2)),
                                                   ReceptorSequence(amino_acid_sequence="AAAACC", metadata=SequenceMetadata(chain="A", count=1)),
                                                   ReceptorSequence(amino_acid_sequence="TADQVF", metadata=SequenceMetadata(chain="A", count=4))],
                                        metadata=RepertoireMetadata(sample=Sample("CD123"), custom_params={"CD": True}))

        sequences = [ReceptorSequence("AAAACA", metadata=SequenceMetadata(chain="A")),
                     ReceptorSequence("TADQV", metadata=SequenceMetadata(chain="A"))]

        matcher = SequenceMatcher()
        result = matcher.match_repertoire(repertoire, 0, sequences, 2, SequenceMatchingSummaryType.COUNT)

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

        result = matcher.match_repertoire(repertoire, 0, sequences, 2, SequenceMatchingSummaryType.CLONAL_PERCENTAGE)
        self.assertEqual(0.8, result["clonal_percentage"])
