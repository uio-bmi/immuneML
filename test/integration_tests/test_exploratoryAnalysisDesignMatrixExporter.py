import os
import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.reference_encoding.ReferenceRepertoireEncoder import ReferenceRepertoireEncoder
from source.encodings.reference_encoding.SequenceMatchingSummaryType import SequenceMatchingSummaryType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.encoding_reports.DesignMatrixExporter import DesignMatrixExporter
from source.util.PathBuilder import PathBuilder
from source.util.RepertoireBuilder import RepertoireBuilder
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class TestExploratoryAnalysisDesignMatrixExporter(TestCase):

    def create_dataset(self, path):

        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                            {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, params={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset

    def test_run(self):

        path = EnvironmentSettings.tmp_test_path + "explanalysisprocintegration/"
        PathBuilder.build(path)
        os.environ["cache_type"] = "test"

        dataset = self.create_dataset(path)

        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                    
        """

        with open(path + "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs = {"path": path + "refs.tsv", "format": "VDJdb"}

        units = {"named_analysis_4": ExploratoryAnalysisUnit(dataset=dataset,
                                                             report=DesignMatrixExporter(),
                                                             label_config=label_config,
                                                             encoder=ReferenceRepertoireEncoder.build_object(dataset,
                                                                                                             **{"max_edit_distance": 1,
                                                                                                "summary": SequenceMatchingSummaryType.COUNT.name,
                                                                                                "reference_sequences": refs}))}

        process = ExploratoryAnalysisInstruction(units)
        process.run(path + "results/")

        self.assertTrue(os.path.isfile(path + "results/analysis_named_analysis_4/design_matrix.csv"))

        shutil.rmtree(path)
