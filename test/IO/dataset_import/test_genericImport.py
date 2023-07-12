import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.GenericImport import GenericImport
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestGenericLoader(TestCase):
    def make_dummy_dataset(self, path, add_metadata):
        rep1text = """Clone ID	Senior Author	TRAJ Gene	TRAV Gene	CDR3A AA Sequence	TRBV Gene	TRBD Gene	TRBJ Gene	CDR3B AA Sequence	Antigen Protein	Antigen Gene	Antigen Species	Antigen Peptide AA #	Epitope Peptide	MHC Class	HLA Restriction	Counts
1E6	Sewell	TRAJ12	TRAV12-3	CAMRGDSSYKLIF	TRBV12-4	TRBD2	TRBJ2-4	CASSLWEKLAKNIQYF	PPI	INS	Human	12-24	ALWGPDPAAA	MHC I	A*02:01	1
4.13	Nepom	TRAJ44	TRAV19	CALSENRGGTASKLTF	TRBV5-1	TRBD1	TRBJ1-1	CASSLVGGPSSEAFF	GAD		Human	555-567		MHC II	DRB1*04:01	3
5	Roep	TRAJ6	TRAV21	CAVKRTGGSYIPTF	TRBV11-2	TRBD1	TRBJ2-2	CASSSFWGSDTGELFF	Insulin B		Human	9-23		MHC II	DQ8	7
D222D 2	Mallone	TRAJ36*01	TRAV17*01	CAVTGANNLFF	TRBV19*01	TRBD1*01	TRBJ2-2*01	CASSIEGPTGELFF	Zinc Transporter 8	ZnT8	Human	185-194	AVAANIVLTV	MHC I	A*02:01	2
GSE.20D11	Nakayama	TRAJ4	TRAV12-3	CAILSGGYNKLIF	TRBV2	TRBD2	TRBJ2-5	CASSAETQYF	Insulin B		Human	9-23		MHC II	DQ8	10
GSE.6H9	Nakayama	TRAJ40	TRAV26-1	CIVRVDSGTYKYIF	TRBV7-2	TRBD2	TRBJ2-1	CASSLTAGLASTYNEQFF	Insulin B		Human	9-23		MHC II	DQ8/DQ8	nan
iGRP 32	DiLorenzo	TRAJ48	TRAV12-1	CVVNILSNFGNEKLTF	TRBV20/OR9-2	TRBD1	TRBJ2-1	CSASRQGWVNEQFF	IGRP		Human	265-273		MHC I	A*02:01	1
MART-1	TBD	TRAJ23	TRAV12-2	CAVNFGGGKLIF	TRBV6-4	TRBD2	TRBJ1-1	CASSLSFGTEAFF	Melan A		Human	27-35	ELAGIGILTV	MHC I	A2	3
MHB10.3	TBD	TRAJ27	TRAV4	CLVGDSLNTNAGKSTF	TRBV29-1	TRBD2	TRBJ2-2	CSVEDRNTGELFF	Insulin B		Human	11-30		MHC II	DRB1*03:01	NA
PM1#11	TBD	TRAJ54	TRAV35	CAGHSIIQGAQKLVF	TRBV5-1	TRBD2	TRBJ2-1	CASGRSSYNEQFF	GAD		Human	339-352		MHC II	DRB1*03:01	2
R164	Nepom	TRAJ56	TRAV19	CALSEEGGGANSKLTF	TRBV5-1	TRBD2	TRBJ1-6	CASSLAGGANSPLHF	GAD		Human	555-567		MHC II	DRB1*04:01	1
SD32.5	Boehm	TRAJ23	TRAV26-1	CIVRVSSAYYNQGGKLIF	TRBV27	TRBD2	TRBJ2-3	CASSPRANTDTQYF	Insulin A		Human	5-21		MHC II	DRB1*04:01	1
SD52.c1	Boehm	TRAJ27	TRAV4	CLVGDSLNTNAGKSTF	TRBV27	TRBD1	TRBJ1-5	CASSWSSIGNQPQHF	PPI	INS	Human	C18-A1		MHC II	DRB1*04:01	1
T1D#10 C8	TBD	TRAJ26	TRAV12-3	CATAYGQNFVF	TRBV4-1	TRBD2	TRBJ2-2	CASSRGGGNTGELFF	Insulin B		Human	9-23		MHC II	DQ8	1
T1D#3 C8	TBD	TRAJ23	TRAV17	CATDAGYNQGGKLIF	TRBV5-1	TRBD2	TRBJ1-3	CASSAGNTIYF	Insulin B		Human	9-23		MHC II	DQ8	1"""
        PathBuilder.remove_old_and_build(path)

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(rep1text)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines(
                    """filename,chain,subject_id,coeliac status (yes/no)
rep1.tsv,TRA,1234e,no"""
                )

    def test_import_repertoire_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "generic_rep"
        self.make_dummy_dataset(path, True)

        dataset = GenericImport.import_dataset({"is_repertoire": True, "result_path": path, "path": path,
                                                "region_type": "IMGT_CDR3", "separator": "\t", "import_empty_aa_sequences": False,
                                                "import_empty_nt_sequences": True,
                                                "column_mapping": {"CDR3B AA Sequence": "sequence_aa", "import_illegal_characters": False,
                                                                   "TRBV Gene": "v_call", "TRBJ Gene": "j_call", "Counts": "duplicate_count"},
                                                "metadata_file": path / "metadata.csv", "number_of_processes": 4}, "generic_dataset")

        self.assertEqual(1, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            self.assertEqual("1234e", rep.metadata["subject_id"])
            self.assertEqual(15, len(rep.sequences))

        repertoire = dataset.get_data()[0]
        self.assertEqual('ASSLWEKLAKNIQY', repertoire.sequences[0].sequence_aa)
        self.assertEqual('ASSLVGGPSSEAF', repertoire.sequences[1].sequence_aa)
        self.assertEqual('ASSSFWGSDTGELF', repertoire.sequences[2].sequence_aa)
        self.assertListEqual([1, 3, 7, 2, 10, None, 1, 3, None, 2, 1, 1, 1, 1, 1], list(rep.get_counts()))
        self.assertListEqual([Chain.BETA for i in range(15)], list(rep.get_chains()))

        shutil.rmtree(path)

    def test_import_sequence_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "generic_seq"
        self.make_dummy_dataset(path, False)

        dataset = GenericImport.import_dataset({"is_repertoire": False, "paired": False,
                                                "result_path": path, "path": path, "import_illegal_characters": False,
                                                "region_type": "IMGT_CDR3", "separator": "\t", "import_empty_nt_sequences": True,
                                                "column_mapping": {"CDR3B AA Sequence": "sequence_aa",
                                                                   "TRBV Gene": "v_call", "TRBJ Gene": "j_call"},
                                                "metadata_file": path / "metadata.csv", "number_of_processes": 4}, "generic_dataset")

        self.assertEqual(15, dataset.get_example_count())

        seqs = [sequence for sequence in dataset.get_data()]

        self.assertEqual('ASSLWEKLAKNIQY', seqs[0].sequence_aa)
        self.assertEqual('ASSLVGGPSSEAF', seqs[1].sequence_aa)
        self.assertEqual('ASSSFWGSDTGELF', seqs[2].sequence_aa)

        shutil.rmtree(path)
