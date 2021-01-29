import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.reference_encoding.MatchedSequencesRepertoireEncoder import MatchedSequencesRepertoireEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.DesignMatrixExporter import DesignMatrixExporter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class TestExploratoryAnalysisDesignMatrixExporter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, labels={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "explanalysisprocintegration/"
        PathBuilder.build(path)
        os.environ["cache_type"] = "test"

        dataset = self.create_dataset(path)

        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV
        """

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs = {"params": {"path": path / "refs.tsv", "region_type": "FULL_SEQUENCE"}, "format": "VDJdb"}

        units = {"named_analysis_4": ExploratoryAnalysisUnit(dataset=dataset,
                                                             report=DesignMatrixExporter(name='report', file_format='csv'),
                                                             label_config=label_config,
                                                             encoder=MatchedSequencesRepertoireEncoder.build_object(dataset,
                                                                                                             **{"max_edit_distance": 1,
                                                                                                                "reference": refs}))}

        process = ExploratoryAnalysisInstruction(units, name="exp")
        process.run(path / "results/")

        self.assertTrue(os.path.isfile(path / "results/exp/analysis_named_analysis_4/report/design_matrix.csv"))

        shutil.rmtree(path)
