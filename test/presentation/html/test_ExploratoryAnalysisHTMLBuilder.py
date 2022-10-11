import os
import shutil
from unittest import TestCase

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.preprocessing.SubjectRepertoireCollector import SubjectRepertoireCollector
from immuneML.presentation.html.ExploratoryAnalysisHTMLBuilder import ExploratoryAnalysisHTMLBuilder
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.reports.encoding_reports.Matches import Matches
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class TestExploratoryAnalysisHTMLBuilder(TestCase):

    def create_dataset(self, path):

        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                            {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, labels={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset

    def test_build(self):
        path = EnvironmentSettings.tmp_test_path / "ea_html_builder"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
                100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                    
                """

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs_dict = {"params": {"path": path / "refs.tsv", "region_type": "FULL_SEQUENCE", "paired": False}, "format": "VDJdb"}

        preproc_sequence = [SubjectRepertoireCollector()]

        encoder = MatchedSequencesEncoder.build_object(dataset, **{
            "reference": refs_dict,
            "max_edit_distance": 0,
            "reads": "all",
            "sum_matches": False,
            "normalize": False
        }, name="test_encoding")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
        ))

        units = {"named_analysis_1": ExploratoryAnalysisUnit(dataset=dataset, report=SequenceLengthDistribution(), number_of_processes=16),
                 "named_analysis_2": ExploratoryAnalysisUnit(dataset=dataset, report=SequenceLengthDistribution(),
                                                             preprocessing_sequence=preproc_sequence,
                                                             label_config=LabelConfiguration(labels=[Label(name="test_label", values=["val1", "val2"], positive_class="val1")])),
                 "named_analysis_3": ExploratoryAnalysisUnit(dataset=encoded, report=Matches(dataset=encoded, name="test_report_name"),
                                                             preprocessing_sequence=preproc_sequence, encoder=encoder,
                                                             label_config=LabelConfiguration())
                 }

        process = ExploratoryAnalysisInstruction(units)
        res = process.run(path / "results/")

        res_path = ExploratoryAnalysisHTMLBuilder.build(res)

        self.assertTrue(os.path.isfile(res_path))

        shutil.rmtree(path)
