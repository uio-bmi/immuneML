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
    def test_get_normalized_sequence_lengths(self):
        path = EnvironmentSettings.tmp_test_path / "tcrmatch"
        PathBuilder.build(path)

        iedb_file = path / "iedb_file"
        self._prepare_iedb_file(iedb_file)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=3,
                                                                     sequence_count_probabilities={100: 0.5,
                                                                                                   120: 0.5},
                                                                     sequence_length_probabilities={12: 0.33,
                                                                                                    14: 0.33,
                                                                                                    15: 0.33},
                                                                     labels={"HLA": {"A": 0.5, "B": 0.5},
                                                                             "CMV": {"+": 0.5, "-": 0.5}, },
                                                                     path=path)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "tcr_match_epitope_analysis")
        params["keep_tmp_results"] = True
        params["dataset"] = dataset
        params["result_path"] = path
        params["iedb_file"] = str(iedb_file)

        compairr_path = Path("/usr/local/bin/compairr")
        tcrmatch_path = Path("/usr/local/bin/tcrmatch")
        params["compairr_path"] = str(compairr_path)
        params["tcrmatch_path"] = str(tcrmatch_path)

        report = TCRMatchEpitopeAnalysis(**params)

        if compairr_path.exists() and tcrmatch_path.exists():
            result = report._generate()

        shutil.rmtree(path)
