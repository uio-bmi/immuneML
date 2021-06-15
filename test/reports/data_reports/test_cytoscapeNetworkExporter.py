import os
import shutil
from unittest import TestCase

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.CytoscapeNetworkExporter import CytoscapeNetworkExporter
from immuneML.util.PathBuilder import PathBuilder


class TestCytoscapeNetworkExporter(TestCase):
    def _create_dummy_data(self, path, dataset_type):
        PathBuilder.build(path)
        dataset = None

        test_repertoire = Repertoire.build(sequence_aas=["DUPDUP", "AILUDGYF", "DFJKHJ", "DIUYUAG", "CTGTCGH"],
                                           v_genes=["V1-1" for i in range(5)],
                                           j_genes=["J1-1" for i in range(5)],
                                           chains=[Chain.ALPHA, Chain.BETA, Chain.BETA, Chain.ALPHA, Chain.BETA],
                                           custom_lists={"custom_1": [f"CUST-{i}" for i in range(5)],
                                                         "custom_2": [f"CUST-A" for i in range(3)] + [f"CUST-B" for i in range(2)]},
                                           cell_ids=[1, 1, 1, 2, 2],
                                           path=path)

        if dataset_type == "receptor":

            dataset = ReceptorDataset.build(test_repertoire.receptors, 100, path, name="receptor_dataset")
            dataset.identifier = 'receptor_dataset'

        elif dataset_type == "repertoire":
            test_repertoire.identifier = "repertoire_dataset"
            dataset = RepertoireDataset(repertoires=[test_repertoire])

        return dataset

    def test_receptor_dataset(self):
        path = EnvironmentSettings.root_path / "test/tmp/cytoscape_export/"
        PathBuilder.build(path)

        receptor_dataset = self._create_dummy_data(path / "data", dataset_type="receptor")

        cne = CytoscapeNetworkExporter(receptor_dataset, path, chains=("alpha", "beta"),
                                       drop_duplicates=True, additional_node_attributes=["custom_1"],
                                       additional_edge_attributes=["custom_2"])

        cne._generate()

        with open(path / "receptor_dataset/all_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=DFJKHJ*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DIUYUAG*v=V1-1*j=J1-1\tpair\t*trb*s=CTGTCGH*v=V1-1*j=J1-1\n'])

        with open(path / "receptor_dataset/shared_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=DFJKHJ*v=V1-1*j=J1-1\n'])

        with open(path / "receptor_dataset/node_metadata.tsv") as file:
            self.assertEqual(file.readline(), 'shared_name\tchain\tsequence\tv_subgroup\tv_gene\tj_subgroup\tj_gene\tcustom_1\tn_duplicates\n')
            self.assertListEqual(sorted(file.readlines()),
                                 sorted(['*tra*s=DUPDUP*v=V1-1*j=J1-1\talpha\tDUPDUP\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-0\t2\n',
                                         '*trb*s=AILUDGYF*v=V1-1*j=J1-1\tbeta\tAILUDGYF\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-1\t1\n',
                                         '*trb*s=DFJKHJ*v=V1-1*j=J1-1\tbeta\tDFJKHJ\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-2\t1\n',
                                         '*tra*s=DIUYUAG*v=V1-1*j=J1-1\talpha\tDIUYUAG\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-3\t1\n',
                                         '*trb*s=CTGTCGH*v=V1-1*j=J1-1\tbeta\tCTGTCGH\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-4\t1\n']))

        with open(path / "receptor_dataset/edge_metadata.tsv") as file:
            self.assertListEqual(file.readlines(),
                                 ['shared_name\tcustom_2\n',
                                  '*tra*s=DUPDUP*v=V1-1*j=J1-1 (pair) *trb*s=AILUDGYF*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*tra*s=DUPDUP*v=V1-1*j=J1-1 (pair) *trb*s=DFJKHJ*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*tra*s=DIUYUAG*v=V1-1*j=J1-1 (pair) *trb*s=CTGTCGH*v=V1-1*j=J1-1\tCUST-B\n'])

        shutil.rmtree(path)

    def test_repertoire_dataset(self):
        path = EnvironmentSettings.root_path / "test/tmp/cytoscape_export/"
        PathBuilder.build(path)

        repertoire_dataset = self._create_dummy_data(path / "data", dataset_type="repertoire")

        cne = CytoscapeNetworkExporter(repertoire_dataset, path, chains=("alpha", "beta"),
                                       drop_duplicates=True, additional_node_attributes=["custom_1"],
                                       additional_edge_attributes=["custom_2"])

        result = cne._generate()

        self.assertIsInstance(result, ReportResult)
        self.assertTrue(os.path.isfile(result.output_tables[0].path))
        self.assertTrue(os.path.isfile(result.output_tables[1].path))
        self.assertTrue(os.path.isfile(result.output_tables[2].path))
        self.assertTrue(os.path.isfile(result.output_tables[3].path))

        with open(path / "repertoire_dataset/all_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=DFJKHJ*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DIUYUAG*v=V1-1*j=J1-1\tpair\t*trb*s=CTGTCGH*v=V1-1*j=J1-1\n']
                                 )

        with open(path / "repertoire_dataset/shared_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*tra*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*trb*s=DFJKHJ*v=V1-1*j=J1-1\n'])

        with open(path / "repertoire_dataset/node_metadata.tsv") as file:
            self.assertEqual(file.readline(), 'shared_name\tchain\tsequence\tv_subgroup\tv_gene\tj_subgroup\tj_gene\tcustom_1\tn_duplicates\n')

            self.assertListEqual(sorted(file.readlines()),
                                 sorted(['*tra*s=DUPDUP*v=V1-1*j=J1-1\talpha\tDUPDUP\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-0\t2\n',
                                         '*trb*s=AILUDGYF*v=V1-1*j=J1-1\tbeta\tAILUDGYF\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-1\t1\n',
                                         '*trb*s=DFJKHJ*v=V1-1*j=J1-1\tbeta\tDFJKHJ\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-2\t1\n',
                                         '*tra*s=DIUYUAG*v=V1-1*j=J1-1\talpha\tDIUYUAG\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-3\t1\n',
                                         '*trb*s=CTGTCGH*v=V1-1*j=J1-1\tbeta\tCTGTCGH\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-4\t1\n']))

        with open(path / "repertoire_dataset/edge_metadata.tsv") as file:
            self.assertListEqual(file.readlines(),
                                 ['shared_name\tcustom_2\n',
                                  '*tra*s=DUPDUP*v=V1-1*j=J1-1 (pair) *trb*s=AILUDGYF*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*tra*s=DUPDUP*v=V1-1*j=J1-1 (pair) *trb*s=DFJKHJ*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*tra*s=DIUYUAG*v=V1-1*j=J1-1 (pair) *trb*s=CTGTCGH*v=V1-1*j=J1-1\tCUST-B\n'])

        shutil.rmtree(path)
