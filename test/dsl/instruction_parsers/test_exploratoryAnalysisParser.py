import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.dsl.instruction_parsers.ExploratoryAnalysisParser import ExploratoryAnalysisParser
from source.encodings.reference_encoding.MatchedReferenceEncoder import MatchedReferenceEncoder
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.reports.encoding_reports.MatchingSequenceDetails import MatchingSequenceDetails
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder


class TestExploratoryAnalysisParser(TestCase):
    def test_parse(self):

        path = EnvironmentSettings.tmp_test_path + "explanalysisparser/"
        PathBuilder.build(path)

        dataset = self.prepare_dataset(path)
        report1 = SequenceLengthDistribution()
        refs = [ReceptorSequence("AAAC", metadata=SequenceMetadata(v_gene="v1", j_gene="j1"))]
        report2 = MatchingSequenceDetails(max_distance=1, reference_sequences=refs)
        encoding = MatchedReferenceEncoder

        instruction = {
            "type": "ExploratoryAnalysis",
            "analyses": [
                {"dataset": "d1", "report": "r1"},
                {"dataset": "d1", "report": "r2", "encoding": "e1", "labels": ["l1"]}
            ]
        }

        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.DATASET, dataset)
        symbol_table.add("r1", SymbolType.REPORT, report1)
        symbol_table.add("r2", SymbolType.REPORT, report2)
        symbol_table.add("e1", SymbolType.ENCODING, encoding, {"encoder_params": {
            "max_edit_distance": 1,
            "summary": SequenceMatchingSummaryType.COUNT,
            "reference_sequences": refs
        }})

        process = ExploratoryAnalysisParser().parse(instruction, symbol_table)

        self.assertEqual(2, len(process.exploratory_analysis_units))
        self.assertTrue(isinstance(process.exploratory_analysis_units[0].report, SequenceLengthDistribution))
        self.assertTrue(isinstance(process.exploratory_analysis_units[1].report, MatchingSequenceDetails))
        self.assertTrue(isinstance(process.exploratory_analysis_units[1].encoder, MatchedReferenceEncoder))
        self.assertEqual(1, len(process.exploratory_analysis_units[1].encoder.reference_sequences))
        self.assertEqual("l1", process.exploratory_analysis_units[1].label_config.get_labels_by_name()[0])

        shutil.rmtree(path)

    def prepare_dataset(self, path: str):
        filenames, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                      {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(filenames=filenames, params={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset
