import shutil
from unittest import TestCase

import pandas as pd
from numpy import nan

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.data_model.SequenceSet import Receptor as TCABReceptor
from immuneML.data_model.SequenceParams import Chain, ChainPair
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestAIRRExporter(TestCase):
    def create_dummy_repertoire(self, path):
        sequence_objects = [ReceptorSequence(sequence_aa="AAA", sequence="GCTGCTGCT", sequence_id="receptor_1",
                                             v_call="TRBV1", j_call="TRBJ1", locus=Chain.BETA.value,
                                             duplicate_count=5, vj_in_frame="T", productive='T',
                                             metadata={"d_call": "TRBD1", "custom_test": "cust1",
                                                       'sig1': 0}),
                            ReceptorSequence(sequence_aa="GGG",
                                             sequence="GGTGGTGGT",
                                             sequence_id="receptor_2",
                                             v_call="TRAV2*01", j_call="TRAJ2", locus=Chain.ALPHA.value,
                                             duplicate_count=15, vj_in_frame="T", productive='F',
                                             metadata={"d_call": "TRAD2", "custom_test": "cust2", 'sig1': 1})]

        repertoire = Repertoire.build_from_sequences(sequences=sequence_objects, result_path=path,
                                                     metadata={"subject_id": "REP1"}, filename_base="REP1")

        dataset = RepertoireDataset.build_from_objects(repertoires=[repertoire], path=path)

        return dataset

    def test_repertoire_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_repertoire/")

        dataset = self.create_dummy_repertoire(path)

        path_exported = path / "exported"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / f"repertoires/{dataset.repertoires[0].metadata['subject_id']}.tsv", sep="\t")

        self.assertListEqual(resulting_data["sequence_id"].to_list(), ["receptor_1", "receptor_2"])
        self.assertListEqual(resulting_data["cdr3"].to_list(), ["GCTGCTGCT", "GGTGGTGGT"])
        self.assertListEqual(resulting_data["cdr3_aa"].to_list(), ["AAA", "GGG"])
        self.assertListEqual(resulting_data["v_call"].to_list(), ["TRBV1", "TRAV2*01"])
        self.assertListEqual(resulting_data["j_call"].to_list(), ["TRBJ1", "TRAJ2"])
        self.assertListEqual(resulting_data["d_call"].to_list(), ["TRBD1", "TRAD2"])
        self.assertListEqual(resulting_data["locus"].to_list(), ["TRB", "TRA"])
        self.assertListEqual(resulting_data["duplicate_count"].to_list(), [5, 15])
        self.assertListEqual(resulting_data["custom_test"].to_list(), ["cust1", "cust2"])
        self.assertListEqual(resulting_data["productive"].to_list(), ['T', 'F'])
        self.assertListEqual(resulting_data["stop_codon"].to_list(), ['F', 'F'])
        self.assertListEqual(resulting_data['sig1'].to_list(), [False, True])

        shutil.rmtree(path)

    def create_dummy_receptordataset(self, path):
        receptors = [TCABReceptor(receptor_id="1", cell_id="1", chain_pair=ChainPair.TRA_TRB,
                                  chain_1=ReceptorSequence(sequence_aa="AAATTT", sequence_id="1a",
                                                           v_call="TRAV1", j_call="TRAJ1", locus=Chain.ALPHA.value,
                                                           vj_in_frame="T", cell_id='1',
                                                           metadata={"d_call": "TRAD1",
                                                                     "custom1": "cust1"}),
                                  chain_2=ReceptorSequence(sequence_aa="ATATAT", sequence_id="1b",
                                                           v_call="TRBV1", j_call="TRBJ1",
                                                           locus=Chain.BETA.value,
                                                           vj_in_frame="T", cell_id='1',
                                                           metadata={"d_call": "TRBD1",
                                                                     "custom1": "cust1"})),
                     TCABReceptor(receptor_id="2", cell_id="2", chain_pair=ChainPair.TRA_TRB,
                                  chain_1=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2a",
                                                           v_call="TRAV1", j_call="TRAJ1",
                                                           locus=Chain.ALPHA.value,
                                                           vj_in_frame="T", cell_id="2",
                                                           metadata={"d_call": "TRAD1",
                                                                     "custom2": "cust1"}),
                                  chain_2=ReceptorSequence(sequence_aa="AAAAAA", sequence_id="2b",
                                                           v_call="TRBV1", j_call="TRBJ1",
                                                           locus=Chain.BETA.value, cell_id="2",
                                                           vj_in_frame="T",
                                                           metadata={"d_call": "TRBD1",
                                                                     "custom2": "cust1"}))]

        receptors_path = path / "receptors"
        PathBuilder.build(receptors_path)
        return ReceptorDataset.build_from_objects(receptors, receptors_path, name='d2')

    def test_receptor_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_receptor/")

        dataset = self.create_dummy_receptordataset(path)

        path_exported = path / "exported_receptors"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / "d2.tsv", sep="\t", dtype=str)

        self.assertListEqual(resulting_data["cell_id"].to_list(), ["1", "1", "2", "2"])
        self.assertListEqual(resulting_data["sequence_id"].to_list(), ["1a", "1b", "2a", "2b"])
        self.assertListEqual(resulting_data["cdr3_aa"].to_list(), ["AAATTT", "ATATAT", "AAAAAA", "AAAAAA"])
        self.assertListEqual(resulting_data["v_call"].to_list(), ["TRAV1", "TRBV1", "TRAV1", "TRBV1"])
        self.assertListEqual(resulting_data["j_call"].to_list(), ["TRAJ1", "TRBJ1", "TRAJ1", "TRBJ1"])
        self.assertListEqual(resulting_data["d_call"].to_list(), ["TRAD1", "TRBD1", "TRAD1", "TRBD1"])
        self.assertListEqual(resulting_data["locus"].to_list(), ["TRA", "TRB", "TRA", "TRB"])
        self.assertListEqual(resulting_data["custom1"].to_list(), ["cust1", "cust1", nan, nan])
        self.assertListEqual(resulting_data["custom2"].to_list(), [nan, nan, "cust1", "cust1"])
        self.assertListEqual(resulting_data["productive"].to_list(), ['T', 'T', 'T', 'T'])
        self.assertListEqual(resulting_data["stop_codon"].to_list(), ['F', 'F', 'F', 'F'])

        shutil.rmtree(path)

    def create_dummy_sequencedataset(self, path):
        sequences = [ReceptorSequence(sequence_aa="AAATTT", sequence_id="1a",
                                      v_call="TRAV1", j_call="TRAJ1", locus=Chain.ALPHA.value,
                                      vj_in_frame="T",
                                      metadata={"d_call": "TRAD1",
                                                "custom1": "cust1"}),
                     ReceptorSequence(sequence_aa="ATATAT", sequence_id="1b",
                                      v_call="TRBV1", j_call="TRBJ1", locus=Chain.BETA.value,
                                      vj_in_frame="T",
                                      metadata={"d_call": "TRBD1",
                                                "custom2": "cust1"}),
                     ReceptorSequence(sequence_aa="ATATAT", sequence_id="2b",
                                      v_call="TRBV1", j_call="TRBJ1", locus=Chain.BETA.value,
                                      vj_in_frame="T",
                                      metadata={"d_call": "TRBD1",
                                                "custom2": "cust1"})]
        sequences_path = path / "sequences"
        PathBuilder.build(sequences_path)
        return SequenceDataset.build_from_objects(sequences, sequences_path, name='d1')

    def test_sequence_export(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "airr_exporter_sequence/")

        dataset = self.create_dummy_sequencedataset(path)

        path_exported = path / "exported_sequences"
        dataset.name = "d1"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported / "d1.tsv", sep="\t", dtype=str, keep_default_na=False)

        self.assertListEqual(resulting_data["sequence_id"].to_list(), ["1a", "1b", "2b"])
        self.assertListEqual(resulting_data["cdr3_aa"].to_list(), ["AAATTT", "ATATAT", "ATATAT"])
        self.assertListEqual(resulting_data["v_call"].to_list(), ["TRAV1", "TRBV1", "TRBV1"])
        self.assertListEqual(resulting_data["j_call"].to_list(), ["TRAJ1", "TRBJ1", "TRBJ1"])
        self.assertListEqual(resulting_data["d_call"].to_list(), ["TRAD1", "TRBD1", "TRBD1"])
        self.assertListEqual(resulting_data["locus"].to_list(), ["TRA", "TRB", "TRB"])
        self.assertListEqual(resulting_data["custom1"].to_list(), ["cust1", '', ''])
        self.assertListEqual(resulting_data["custom2"].to_list(), ['', "cust1", "cust1"])
        self.assertListEqual(resulting_data["productive"].to_list(), ['T', 'T', 'T'])
        self.assertListEqual(resulting_data["stop_codon"].to_list(), ['F', 'F', 'F'])

        shutil.rmtree(path)
