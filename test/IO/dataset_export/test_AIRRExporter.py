import shutil
from unittest import TestCase

import pandas as pd
from numpy import nan

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestAIRRExporter(TestCase):
    def create_dummy_repertoire(self, path):
        sequence_objects = [ReceptorSequence(amino_acid_sequence="AAA",
                                             nucleotide_sequence="GCTGCTGCT",
                                             identifier="receptor_1",
                                             metadata=SequenceMetadata(v_call="TRBV1",
                                                                       j_call="TRBJ1",
                                                                       chain=Chain.BETA,
                                                                       duplicate_count=5,
                                                                       region_type="IMGT_CDR3",
                                                                       frame_type="IN",
                                                                       custom_params={"d_call": "TRBD1",
                                                                                      "custom_test": "cust1",
                                                                                      'sig1': 0})),
                            ReceptorSequence(amino_acid_sequence="GGG",
                                             nucleotide_sequence="GGTGGTGGT",
                                             identifier="receptor_2",
                                             metadata=SequenceMetadata(v_call="TRAV2*01",
                                                                       j_call="TRAJ2",
                                                                       chain=Chain.ALPHA,
                                                                       duplicate_count=15,
                                                                       frame_type=SequenceFrameType.UNDEFINED,
                                                                       region_type="IMGT_CDR3",
                                                                       custom_params={"d_call": "TRAD2",
                                                                                      "custom_test": "cust2",
                                                                                      'sig1': 1}))]

        repertoire = Repertoire.build_from_sequence_objects(sequence_objects=sequence_objects, path=path, metadata={"subject_id": "REP1"})
        df = pd.DataFrame({"filename": [f"{repertoire.identifier}_data.npy"], "subject_id": ["REP1"],
                           "repertoire_identifier": [repertoire.identifier]})
        df.to_csv(path / "metadata.csv", index=False)

        return repertoire, path / "metadata.csv"

    def test_repertoire_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_repertoire/")

        repertoire, metadata_path = self.create_dummy_repertoire(path)
        dataset = RepertoireDataset(repertoires=[repertoire], metadata_file=metadata_path)

        path_exported = path / "exported"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / f"repertoires/{repertoire.metadata['subject_id']}.tsv", sep="\t")

        self.assertListEqual(list(resulting_data["sequence_id"]), ["receptor_1", "receptor_2"])
        self.assertListEqual(list(resulting_data["cdr3"]), ["GCTGCTGCT", "GGTGGTGGT"])
        self.assertListEqual(list(resulting_data["cdr3_aa"]), ["AAA", "GGG"])
        self.assertListEqual(list(resulting_data["v_call"]), ["TRBV1", "TRAV2*01"])
        self.assertListEqual(list(resulting_data["j_call"]), ["TRBJ1", "TRAJ2"])
        self.assertListEqual(list(resulting_data["d_call"]), ["TRBD1", "TRAD2"])
        self.assertListEqual(list(resulting_data["locus"]), ["TRB", "TRA"])
        self.assertListEqual(list(resulting_data["duplicate_count"]), [5, 15])
        self.assertListEqual(list(resulting_data["custom_test"]), ["cust1", "cust2"])
        self.assertListEqual(list(resulting_data["productive"]), ['T', 'F'])
        self.assertListEqual(list(resulting_data["stop_codon"]), ['F', 'F'])
        self.assertListEqual(list(resulting_data['sig1']), [False, True])

        shutil.rmtree(path)

    def create_dummy_receptordataset(self, path):
        receptors = [TCABReceptor(identifier="1",
                                  alpha=ReceptorSequence(amino_acid_sequence="AAATTT", identifier="1a",
                                                         metadata=SequenceMetadata(v_call="TRAV1", j_call="TRAJ1",
                                                                                   chain=Chain.ALPHA,
                                                                                   frame_type="IN",
                                                                                   region_type="IMGT_CDR3",
                                                                                   custom_params={"d_call": "TRAD1",
                                                                                                  "custom1": "cust1"})),
                                  beta=ReceptorSequence(amino_acid_sequence="ATATAT", identifier="1b",
                                                        metadata=SequenceMetadata(v_call="TRBV1", j_call="TRBJ1",
                                                                                  chain=Chain.BETA,
                                                                                  frame_type="IN",
                                                                                  region_type="IMGT_CDR3",
                                                                                  custom_params={"d_call": "TRBD1",
                                                                                                 "custom1": "cust1"}))),
                     TCABReceptor(identifier="2",
                                  alpha=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2a",
                                                         metadata=SequenceMetadata(v_call="TRAV1", j_call="TRAJ1",
                                                                                   chain=Chain.ALPHA,
                                                                                   frame_type="IN",
                                                                                   region_type="IMGT_CDR3",
                                                                                   custom_params={"d_call": "TRAD1",
                                                                                                  "custom2": "cust1"})),
                                  beta=ReceptorSequence(amino_acid_sequence="AAAAAA", identifier="2b",
                                                        metadata=SequenceMetadata(v_call="TRBV1", j_call="TRBJ1",
                                                                                  chain=Chain.BETA,
                                                                                  frame_type="IN",
                                                                                  region_type="IMGT_CDR3",
                                                                                  custom_params={"d_call": "TRBD1",
                                                                                                 "custom2": "cust1"})))]

        receptors_path = path / "receptors"
        PathBuilder.build(receptors_path)
        return ReceptorDataset.build_from_objects(receptors, 2, receptors_path)

    def test_receptor_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_receptor/")

        dataset = self.create_dummy_receptordataset(path)

        path_exported = path / "exported_receptors"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / "batch1.tsv", sep="\t", dtype=str)

        self.assertListEqual(list(resulting_data["cell_id"]), ["1", "1", "2", "2"])
        self.assertListEqual(list(resulting_data["sequence_id"]), ["1a", "1b", "2a", "2b"])
        self.assertListEqual(list(resulting_data["cdr3_aa"]), ["AAATTT", "ATATAT", "AAAAAA", "AAAAAA"])
        self.assertListEqual(list(resulting_data["v_call"]), ["TRAV1", "TRBV1", "TRAV1", "TRBV1"])
        self.assertListEqual(list(resulting_data["j_call"]), ["TRAJ1", "TRBJ1", "TRAJ1", "TRBJ1"])
        self.assertListEqual(list(resulting_data["d_call"]), ["TRAD1", "TRBD1", "TRAD1", "TRBD1"])
        self.assertListEqual(list(resulting_data["locus"]), ["TRA", "TRB", "TRA", "TRB"])
        self.assertListEqual(list(resulting_data["custom1"]), ["cust1", "cust1", nan, nan])
        self.assertListEqual(list(resulting_data["custom2"]), [nan, nan, "cust1", "cust1"])
        self.assertListEqual(list(resulting_data["productive"]), ['T', 'T', 'T', 'T'])
        self.assertListEqual(list(resulting_data["stop_codon"]), ['F', 'F', 'F', 'F'])

        shutil.rmtree(path)

    def create_dummy_sequencedataset(self, path):
        sequences = [ReceptorSequence(amino_acid_sequence="AAATTT", identifier="1a",
                                      metadata=SequenceMetadata(v_call="TRAV1", j_call="TRAJ1", chain=Chain.ALPHA, frame_type="IN",
                                                                region_type="IMGT_CDR3",
                                                                custom_params={"d_call": "TRAD1",
                                                                               "custom1": "cust1"})),
                     ReceptorSequence(amino_acid_sequence="ATATAT", identifier="1b",
                                      metadata=SequenceMetadata(v_call="TRBV1", j_call="TRBJ1", chain=Chain.BETA, frame_type="IN",
                                                                region_type="IMGT_CDR3",
                                                                custom_params={"d_call": "TRBD1",
                                                                               "custom2": "cust1"})),
                     ReceptorSequence(amino_acid_sequence="ATATAT", identifier="2b",
                                      metadata=SequenceMetadata(v_call="TRBV1", j_call="TRBJ1", chain=Chain.BETA, frame_type="IN",
                                                                region_type="IMGT_CDR3",
                                                                custom_params={"d_call": "TRBD1",
                                                                               "custom2": "cust1"}))]
        sequences_path = path / "sequences"
        PathBuilder.build(sequences_path)
        return SequenceDataset.build_from_objects(sequences, 2, sequences_path)

    def test_sequence_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_sequence/")

        dataset = self.create_dummy_sequencedataset(path)

        path_exported = path / "exported_sequences"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / "batch1.tsv", sep="\t")

        self.assertListEqual(list(resulting_data["sequence_id"]), ["1a", "1b"])
        self.assertListEqual(list(resulting_data["cdr3_aa"]), ["AAATTT", "ATATAT"])
        self.assertListEqual(list(resulting_data["v_call"]), ["TRAV1", "TRBV1"])
        self.assertListEqual(list(resulting_data["j_call"]), ["TRAJ1", "TRBJ1"])
        self.assertListEqual(list(resulting_data["d_call"]), ["TRAD1", "TRBD1"])
        self.assertListEqual(list(resulting_data["locus"]), ["TRA", "TRB"])
        self.assertListEqual(list(resulting_data["custom1"]), ["cust1", nan])
        self.assertListEqual(list(resulting_data["custom2"]), [nan, "cust1"])
        self.assertListEqual(list(resulting_data["productive"]), ['T', 'T'])
        self.assertListEqual(list(resulting_data["stop_codon"]), ['F', 'F'])

        resulting_data = pd.read_csv(path_exported / "batch2.tsv", sep="\t")
        self.assertListEqual(list(resulting_data["sequence_id"]), ["2b"])
        self.assertListEqual(list(resulting_data["cdr3_aa"]), ["ATATAT"])
        self.assertListEqual(list(resulting_data["v_call"]), ["TRBV1"])
        self.assertListEqual(list(resulting_data["j_call"]), ["TRBJ1"])
        self.assertListEqual(list(resulting_data["d_call"]), ["TRBD1"])
        self.assertListEqual(list(resulting_data["locus"]), ["TRB"])
        self.assertListEqual(list(resulting_data["custom2"]), ["cust1"])
        self.assertListEqual(list(resulting_data["productive"]), ['T'])
        self.assertListEqual(list(resulting_data["stop_codon"]), ['F'])

        shutil.rmtree(path)
