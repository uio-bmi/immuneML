import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.dsl.instruction_parsers.ExploratoryAnalysisParser import ExploratoryAnalysisParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.SubjectRepertoireCollector import SubjectRepertoireCollector
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.reports.encoding_reports.Matches import Matches
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestExploratoryAnalysisParser(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_parse(self):

        path = EnvironmentSettings.tmp_test_path / "explanalysisparser/"
        PathBuilder.remove_old_and_build(path)

        dataset = self.prepare_dataset(path)
        report1 = SequenceLengthDistribution()

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV
        """

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs = {"params": {"path": path / "refs.tsv", "region_type": "FULL_SEQUENCE"}, "format": "VDJdb"}

        report2 = Matches.build_object()
        encoding = MatchedSequencesEncoder
        p1 = [SubjectRepertoireCollector()]

        instruction = {
            "type": "ExploratoryAnalysis",
            "number_of_processes": 32,
            "analyses": {
                "1": {"dataset": "d1", "report": "r1", "preprocessing_sequence": "p1"},
                "2": {"dataset": "d1", "report": "r2", "encoding": "e1", },
                "3": {"dataset": "d1", "report": "r2", "encoding": "e1", "labels": ["l1"]}
            }
        }

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)
        symbol_table.add("r1", SymbolType.REPORT, report1)
        symbol_table.add("r2", SymbolType.REPORT, report2)
        symbol_table.add("e1", SymbolType.ENCODING, encoding, {"encoder_params": {
            "max_edit_distance": 1,
            "reference": refs,
            "reads": "all",
            "sum_matches": False,
            "normalize": False
        }})
        symbol_table.add("p1", SymbolType.PREPROCESSING, p1)

        process = ExploratoryAnalysisParser().parse("a", instruction, symbol_table)

        self.assertEqual(3, len(list(process.state.exploratory_analysis_units.values())))
        self.assertTrue(isinstance(list(process.state.exploratory_analysis_units.values())[0].report, SequenceLengthDistribution))

        # testing matches with and without labels
        self.assertTrue(isinstance(list(process.state.exploratory_analysis_units.values())[1].report, Matches))
        self.assertTrue(isinstance(list(process.state.exploratory_analysis_units.values())[1].encoder, MatchedSequencesEncoder))
        self.assertEqual(1, len(list(process.state.exploratory_analysis_units.values())[1].encoder.reference_sequences))

        self.assertTrue(isinstance(list(process.state.exploratory_analysis_units.values())[2].report, Matches))
        self.assertTrue(isinstance(list(process.state.exploratory_analysis_units.values())[2].encoder, MatchedSequencesEncoder))
        self.assertEqual(1, len(list(process.state.exploratory_analysis_units.values())[2].encoder.reference_sequences))
        self.assertEqual("l1", list(process.state.exploratory_analysis_units.values())[2].label_config.get_labels_by_name()[0])
        self.assertEqual(32, process.state.exploratory_analysis_units["2"].number_of_processes)

        shutil.rmtree(path)

    def prepare_dataset(self, path: str):
        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, labels={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset
