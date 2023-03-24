import os
import shutil
from pathlib import Path
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.TCRMatchEpitopeAnalysis import TCRMatchEpitopeAnalysis
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_iedb_file(self, filename):
        with open(filename, "w") as file:
            file.writelines(["cdr3_aa	original_seq	receptor_group	epitopes	source_organisms	source_antigens\n",
                             "ASSQDRDTQY	ASSQDRDTQY	47	VMAPRTLIL	Human herpesvirus 5 (Human cytomegalovirus)	glycoprotein	\n",
                             "ASGDAGGGYEQY	ASGDAGGGYEQY	8606	ASQKRPSQRSK,ASQKRPSQRHG,ASQKRPSQR,ASQARPSQR,AGQARPSQR,ASQYRPSQR,ASQFRPSQR,AGQFRPSQR	Mus musculus (mouse),Rattus norvegicus (brown rat),Mus musculus (mouse)	Myelin basic protein,myelin basic protein,myelin basic protein	\n",
                             "ASGDAGGGYEQY	ASGDAGGGYEQYFGP	18226	ASQKRPSQR	Mus musculus (mouse)	Myelin basic protein	\n"])

    def _prepare_tcrmatch_output_files(self, path, identifiers):
        PathBuilder.build(path)

        assert len(identifiers) == 4
        filepaths = [path / f"{i}.tsv" for i in identifiers]

        with open(filepaths[0], "w") as file:
            file.writelines(["input_sequence	match_sequence	score	receptor_group	epitope	antigen	organism\n",
                             "ASSQEFGAGLQLETQY	ASSQEFGAGLQLETQY	1.0000	119379	HTTDPSFLGRY	orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSPPPGTYSYEQY	ASSPPPGTYSYEQY	1.0000	110991	MIELSLIDFYLCFLAFLLFLVLIML	ORF7b [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSDSGYSYNEQF	ASSDSGYSYNEQF	1.0000	80456	AELAKNVSLDNVL	orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n"])

        with open(filepaths[1], "w") as file:
            file.writelines(["input_sequence	match_sequence	score	receptor_group	epitope	antigen	organism\n",
                             "ASSQEFGAGLQLETQY	ASSQEFGAGLQLETQY	1.0000	119379	HTTDPSFLGRY	orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSPPPGTYSYEQY	ASSPPPGTYSYEQY	1.0000	110991	MIELSLIDFYLCFLAFLLFLVLIML	ORF7b [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ATSDLSTGDHDTQY	ATSDLSTGDHDTQY	1.0000	140435	LSPRWYFYYL	nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n"])

        with open(filepaths[2], "w") as file:
            file.writelines(["input_sequence	match_sequence	score	receptor_group	epitope	antigen	organism\n",
                             "ASSQEFGAGLQLETQY	ASSQEFGAGLQLETQY	1.0000	119379	HTTDPSFLGRY	orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ATSDLSTGDHDTQY	ATSDLSTGDHDTQY	1.0000	140435	LSPRWYFYYL	nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSEQEGYSSLNQPQH	ASSEQEGYSSLNQPQH	1.0000	79527	KLPDDFTGCV	surface glycoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n"])

        with open(filepaths[3], "w") as file:
            file.writelines(["input_sequence	match_sequence	score	receptor_group	epitope	antigen	organism\n",
                             "ATSDLSTGDHDTQY	ATSDLSTGDHDTQY	1.0000	140435	LSPRWYFYYL	nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSEQEGYSSLNQPQH	ASSEQEGYSSLNQPQH	1.0000	79527	KLPDDFTGCV	surface glycoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n",
                             "ASSHGRGPNQPQH	ASSHGRGPNQPQH	1.0000	86004	LSPRWYFYYL	nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2]	SARS-CoV2\n"])

        return filepaths

    def test_get_normalized_sequence_lengths(self):
        path = EnvironmentSettings.tmp_test_path / "tcrmatch"
        PathBuilder.build(path)

        iedb_file = path / "iedb_file"
        self._prepare_iedb_file(iedb_file)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=4,
                                                                     sequence_count_probabilities={100: 0.5,
                                                                                                   120: 0.5},
                                                                     sequence_length_probabilities={12: 0.33,
                                                                                                    14: 0.33,
                                                                                                    15: 0.33},
                                                                     labels={"HLA": {"A": 0.5, "B": 0.5},
                                                                             "CMV": {"1": 0.5, "0": 0.5}},
                                                                     path=path,
                                                                     random_seed=2)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "tcr_match_epitope_analysis")
        params["keep_tmp_results"] = True
        params["dataset"] = dataset
        params["result_path"] = path
        params["iedb_file"] = str(iedb_file)

        report = TCRMatchEpitopeAnalysis(**params)

        # result = report._generate()

        tcrmatch_files = self._prepare_tcrmatch_output_files(path / "tcrmatch_results",
                                                                 identifiers=dataset.get_repertoire_ids())

        result = report._generate_report_results(tcrmatch_files)

        self.assertTrue(os.path.isfile(path / "tcrmatch_per_repertoire.tsv"))

        self.assertTrue(os.path.isfile(path / "label=CMV_organism=SARS-CoV2_antigen=nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=CMV_organism=SARS-CoV2_antigen=orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=CMV_organism=SARS-CoV2_antigen=ORF7b [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=CMV_organism=SARS-CoV2_antigen=surface glycoprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=CMV_tcrmatch_summary.tsv"))
        self.assertTrue(os.path.isfile(path / "tcrmatch_summary_label=CMV_0_vs_1.html"))

        self.assertTrue(os.path.isfile(path / "label=HLA_organism=SARS-CoV2_antigen=nucleocapsid phosphoprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=HLA_organism=SARS-CoV2_antigen=orf1ab polyprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=HLA_organism=SARS-CoV2_antigen=ORF7b [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=HLA_organism=SARS-CoV2_antigen=surface glycoprotein [Severe acute respiratory syndrome coronavirus 2].html"))
        self.assertTrue(os.path.isfile(path / "label=HLA_tcrmatch_summary.tsv"))

        shutil.rmtree(path)
