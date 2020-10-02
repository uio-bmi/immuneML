import shutil
from unittest import TestCase

import pandas as pd

from source.IO.dataset_export.AIRRExporter import AIRRExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestAIRRExporter(TestCase):
    def create_dummy_repertoire(self, path):
        sequence_objects = [ReceptorSequence(amino_acid_sequence="AAA",
                                             nucleotide_sequence="GCTGCTGCT",
                                             identifier="receptor_1",
                                             metadata=SequenceMetadata(v_gene="v1",
                                                                       j_gene="j1",
                                                                       chain=Chain.BETA,
                                                                       count=5,
                                                                       region_type="CDR3",
                                                                       custom_params={"d_call": "d1",
                                                                                      "custom_test": "cust1"})),
                            ReceptorSequence(amino_acid_sequence="GGG",
                                             nucleotide_sequence="GGTGGTGGT",
                                             identifier="receptor_2",
                                             metadata=SequenceMetadata(v_gene="v2",
                                                                       j_gene="j2",
                                                                       chain=Chain.ALPHA,
                                                                       count=15,
                                                                       region_type="CDR3",
                                                                       custom_params={"d_call": "d2",
                                                                                      "custom_test": "cust2"}))]

        repertoire = Repertoire.build_from_sequence_objects(sequence_objects=sequence_objects, path=path, metadata={"subject_id": "REP1"})
        df = pd.DataFrame({"filename": [f"{repertoire.identifier}_data.npy"], "subject_id": ["1"],
                           "repertoire_identifier": [repertoire.identifier]})
        df.to_csv(path + "metadata.csv", index=False)

        return repertoire, path + "metadata.csv"

    def test_export(self):
        path = EnvironmentSettings.tmp_test_path + "airr_exporter/"
        PathBuilder.build(path)

        repertoire, metadata_path = self.create_dummy_repertoire(path)
        dataset = RepertoireDataset(repertoires=[repertoire], metadata_file=metadata_path)

        path_exported = f"{path}exported/"
        AIRRExporter.export(dataset, path_exported)

        resulting_data = pd.read_csv(path_exported + f"repertoires/{repertoire.identifier}.tsv", sep="\t")

        self.assertListEqual(list(resulting_data["sequence_id"]), ["receptor_1", "receptor_2"])
        self.assertListEqual(list(resulting_data["cdr3"]), ["GCTGCTGCT", "GGTGGTGGT"])
        self.assertListEqual(list(resulting_data["cdr3_aa"]), ["AAA", "GGG"])
        self.assertListEqual(list(resulting_data["v_call"]), ["v1", "v2"])
        self.assertListEqual(list(resulting_data["j_call"]), ["j1", "j2"])
        self.assertListEqual(list(resulting_data["d_call"]), ["d1", "d2"])
        self.assertListEqual(list(resulting_data["locus"]), ["TRB", "TRA"])
        self.assertListEqual(list(resulting_data["duplicate_count"]), [5, 15])
        self.assertListEqual(list(resulting_data["custom_test"]), ["cust1", "cust2"])

        shutil.rmtree(EnvironmentSettings.tmp_test_path + "airr_exporter/")
