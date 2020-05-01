import os
import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.reference_encoding.ReferenceRepertoireEncoder import ReferenceRepertoireEncoder
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.preprocessing.PatientRepertoireCollector import PatientRepertoireCollector
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.reports.encoding_reports.MatchingSequenceDetails import MatchingSequenceDetails
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class TestExploratoryAnalysisProcess(TestCase):

    def create_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, params={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path + "explanalysisproc/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                    
        """

        with open(path + "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs_dict = {"path": path + "refs.tsv", "format": "VDJdb"}

        preproc_sequence = [PatientRepertoireCollector()]

        units = {"named_analysis_1": ExploratoryAnalysisUnit(dataset=dataset, report=SequenceLengthDistribution(), batch_size=16),
                 "named_analysis_2": ExploratoryAnalysisUnit(dataset=dataset, report=SequenceLengthDistribution(),
                                                             preprocessing_sequence=preproc_sequence),
                 "named_analysis_3": ExploratoryAnalysisUnit(dataset=dataset,
                                                             report=MatchingSequenceDetails.build_object(max_edit_distance=1,
                                                                                                         reference_sequences=refs_dict),
                                                             label_config=label_config,
                                                             encoder=ReferenceRepertoireEncoder.build_object(dataset,
                                                                                                             **{"max_edit_distance": 1,
                                                                                                                "summary": SequenceMatchingSummaryType.COUNT.name,
                                                                                                                "reference_sequences": refs_dict}))}

        process = ExploratoryAnalysisInstruction(units)
        process.run(path + "results/")

        self.assertTrue(units["named_analysis_1"].batch_size == 16)
        self.assertTrue(os.path.isfile(path + "results/analysis_named_analysis_1/sequence_length_distribution.png"))
        self.assertTrue(os.path.isfile(path + "results/analysis_named_analysis_2/sequence_length_distribution.png"))
        self.assertTrue(os.path.isfile(path + "results/analysis_named_analysis_3/matching_sequence_overview.tsv"))

        shutil.rmtree(path)
