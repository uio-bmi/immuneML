import os
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.split_dataset.SplitDatasetInstruction import SplitDatasetState


class SplitDatasetHTMLBuilder:
    """
    A class that builds HTML file(s) from a SplitDatasetState object to display
    the results of the SplitDataset instruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: SplitDatasetState) -> Path:
        """
        Build the HTML files based on the SplitDataset state.

        Arguments:
            state: SplitDatasetState object including all details of the SplitDataset instruction

        Returns:
            Path to the main HTML file (located under state.result_path)
        """
        base_path = PathBuilder.build((state.result_path / "../HTML_output/"))
        html_map = SplitDatasetHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"SplitDataset_{state.name}.html"

        TemplateParser.parse(
            template_path=EnvironmentSettings.html_templates_path / "SplitDataset.html",
            template_map=html_map,
            result_path=result_file
        )

        return result_file

    @staticmethod
    def make_html_map(state: SplitDatasetState, base_path: Path) -> dict:
        train_zip_path = None
        test_zip_path = None

        if state.train_data_path:
            zip_file = Util.make_downloadable_zip(state.result_path, state.train_data_path, "train_dataset")
            train_zip_path = os.path.relpath(zip_file, base_path)

        if state.test_data_path:
            zip_file = Util.make_downloadable_zip(state.result_path, state.test_data_path, "test_dataset")
            test_zip_path = os.path.relpath(zip_file, base_path)

        html_map = {
            "css_style": Util.get_css_content(SplitDatasetHTMLBuilder.CSS_PATH),
            "name": state.name,
            "immuneML_version": MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "split_strategy": state.split_config.split_strategy.name.lower(),
            "training_percentage": f"{state.split_config.training_percentage * 100:.0f}%"
                if state.split_config.training_percentage else "N/A",
            "test_percentage": f"{(1 - state.split_config.training_percentage) * 100:.0f}%"
                if state.split_config.training_percentage else "N/A",
            "train_dataset_download_link": train_zip_path,
            "test_dataset_download_link": test_zip_path,
            **Util.make_dataset_html_map(state.dataset, dataset_key="original_dataset")
        }

        return html_map