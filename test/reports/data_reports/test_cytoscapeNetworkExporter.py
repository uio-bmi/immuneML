import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.CytoscapeNetworkExporter import CytoscapeNetworkExporter
from source.util.PathBuilder import PathBuilder


class TestCytoscapeNetworkExporter(TestCase):
    def _create_dummy_data(self, path, dataset_type):
        PathBuilder.build(path)

        test_repertoire = Repertoire.build(sequence_aas=["DUPDUP", "AILUDGYF", "DFJKHJ", "DIUYUAG", "CTGTCGH"],
                                           v_genes=["V1-1" for i in range(5)],
                                           j_genes=["J1-1" for i in range(5)],
                                           chains=[Chain.A, Chain.B, Chain.B, Chain.A, Chain.B],
                                           custom_lists={"custom_1": [f"CUST-{i}" for i in range(5)],
                                                         "custom_2": [f"CUST-A" for i in range(3)] + [f"CUST-B" for i in range(2)]},
                                           cell_ids=[1, 1, 1, 2, 2],
                                           path=path)

        if dataset_type == "receptor":
            receptordataset_filename = f"{path}/receptors.pkl"
            with open(receptordataset_filename, "wb") as file:
                pickle.dump(test_repertoire.receptors, file)

            dataset = ReceptorDataset(filenames=[receptordataset_filename], identifier="receptor_dataset")

        elif dataset_type == "repertoire":
            test_repertoire.identifier = "repertoire_dataset"
            dataset = RepertoireDataset(repertoires=[test_repertoire])

        return dataset

    def test_receptor_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/cytoscape_export/"
        PathBuilder.build(path)

        receptor_dataset = self._create_dummy_data(f"{path}/data/", dataset_type="receptor")

        cne = CytoscapeNetworkExporter(receptor_dataset, path, chains=("alpha", "beta"),
                                       drop_duplicates=True, additional_node_attributes=["custom_1"],
                                       additional_edge_attributes=["custom_2"])

        cne.generate()

        with open(f"{path}receptor_dataset/all_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=DFJKHJ*v=V1-1*j=J1-1\n',
                                                    '*a*s=DIUYUAG*v=V1-1*j=J1-1\tpair\t*b*s=CTGTCGH*v=V1-1*j=J1-1\n']
)

        with open(f"{path}receptor_dataset/shared_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=DFJKHJ*v=V1-1*j=J1-1\n'])

        with open(f"{path}receptor_dataset/node_metadata.tsv") as file:
            self.assertEqual(file.readline(), 'shared_name\tchain\tsequence\tv_subgroup\tv_gene\tj_subgroup\tj_gene\tcustom_1\tn_duplicates\n')
            self.assertListEqual(sorted(file.readlines()), sorted(['*a*s=DUPDUP*v=V1-1*j=J1-1\talpha\tDUPDUP\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-0\t2\n',
                                                    '*b*s=AILUDGYF*v=V1-1*j=J1-1\tbeta\tAILUDGYF\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-1\t1\n',
                                                    '*b*s=DFJKHJ*v=V1-1*j=J1-1\tbeta\tDFJKHJ\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-2\t1\n',
                                                    '*a*s=DIUYUAG*v=V1-1*j=J1-1\talpha\tDIUYUAG\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-3\t1\n',
                                                    '*b*s=CTGTCGH*v=V1-1*j=J1-1\tbeta\tCTGTCGH\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-4\t1\n']))

        with open(f"{path}receptor_dataset/edge_metadata.tsv") as file:
            self.assertListEqual(file.readlines(),
                                 ['shared_name\tcustom_2\n',
                                  '*a*s=DUPDUP*v=V1-1*j=J1-1 (pair) *b*s=AILUDGYF*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*a*s=DUPDUP*v=V1-1*j=J1-1 (pair) *b*s=DFJKHJ*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*a*s=DIUYUAG*v=V1-1*j=J1-1 (pair) *b*s=CTGTCGH*v=V1-1*j=J1-1\tCUST-B\n'])

        shutil.rmtree(path)

    def test_repertoire_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/cytoscape_export/"
        PathBuilder.build(path)

        repertoire_dataset = self._create_dummy_data(f"{path}/data/", dataset_type="repertoire")

        cne = CytoscapeNetworkExporter(repertoire_dataset, path, chains=("alpha", "beta"),
                                       drop_duplicates=True, additional_node_attributes=["custom_1"],
                                       additional_edge_attributes=["custom_2"])

        cne.generate()

        with open(f"{path}repertoire_dataset/all_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=DFJKHJ*v=V1-1*j=J1-1\n',
                                                    '*a*s=DIUYUAG*v=V1-1*j=J1-1\tpair\t*b*s=CTGTCGH*v=V1-1*j=J1-1\n']
                                 )

        with open(f"{path}repertoire_dataset/shared_chains.sif") as file:
            self.assertListEqual(file.readlines(), ['*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=AILUDGYF*v=V1-1*j=J1-1\n',
                                                    '*a*s=DUPDUP*v=V1-1*j=J1-1\tpair\t*b*s=DFJKHJ*v=V1-1*j=J1-1\n'])

        with open(f"{path}repertoire_dataset/node_metadata.tsv") as file:
            self.assertEqual(file.readline(), 'shared_name\tchain\tsequence\tv_subgroup\tv_gene\tj_subgroup\tj_gene\tcustom_1\tn_duplicates\n')

            self.assertListEqual(sorted(file.readlines()),
                                 sorted(['*a*s=DUPDUP*v=V1-1*j=J1-1\talpha\tDUPDUP\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-0\t2\n',
                                  '*b*s=AILUDGYF*v=V1-1*j=J1-1\tbeta\tAILUDGYF\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-1\t1\n',
                                  '*b*s=DFJKHJ*v=V1-1*j=J1-1\tbeta\tDFJKHJ\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-2\t1\n',
                                  '*a*s=DIUYUAG*v=V1-1*j=J1-1\talpha\tDIUYUAG\tTRAV1\tTRAV1-1\tTRAJ1\tTRAJ1-1\tCUST-3\t1\n',
                                  '*b*s=CTGTCGH*v=V1-1*j=J1-1\tbeta\tCTGTCGH\tTRBV1\tTRBV1-1\tTRBJ1\tTRBJ1-1\tCUST-4\t1\n']))

        with open(f"{path}repertoire_dataset/edge_metadata.tsv") as file:
            self.assertListEqual(file.readlines(),
                                 ['shared_name\tcustom_2\n',
                                  '*a*s=DUPDUP*v=V1-1*j=J1-1 (pair) *b*s=AILUDGYF*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*a*s=DUPDUP*v=V1-1*j=J1-1 (pair) *b*s=DFJKHJ*v=V1-1*j=J1-1\tCUST-A\n',
                                  '*a*s=DIUYUAG*v=V1-1*j=J1-1 (pair) *b*s=CTGTCGH*v=V1-1*j=J1-1\tCUST-B\n'])

        shutil.rmtree(path)
