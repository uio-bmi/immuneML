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
        training_percentage = state.split_config.training_percentage
        has_percentage = training_percentage is not None and training_percentage > 0
        multiple_splits = len(state.train_data_paths) > 1

        splits = []
        for index, (train_path, test_path) in enumerate(zip(state.train_data_paths, state.test_data_paths)):
            suffix = f"_split_{index + 1}" if multiple_splits else ""
            train_zip = Util.make_downloadable_zip(state.result_path, train_path, f"train_dataset{suffix}")
            test_zip = Util.make_downloadable_zip(state.result_path, test_path, f"test_dataset{suffix}")
            splits.append({
                "split_name": f"Split {index + 1}" if multiple_splits else "",
                "train_dataset_download_link": os.path.relpath(train_zip, base_path),
                "test_dataset_download_link": os.path.relpath(test_zip, base_path),
                "train_dataset_file": Path(train_zip).name,
                "test_dataset_file": Path(test_zip).name,
            })

        group_config = state.split_config.group_k_fold_config

        html_map = {
            "css_style": Util.get_css_content(SplitDatasetHTMLBuilder.CSS_PATH),
            "name": state.name,
            "immuneML_version": MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "split_strategy": state.split_config.split_strategy.name.lower(),
            "split_count": state.split_config.split_count,
            "training_percentage": f"{training_percentage * 100:.0f}%" if has_percentage else "N/A",
            "test_percentage": f"{(1 - training_percentage) * 100:.0f}%" if has_percentage else "N/A",
            "show_group_config": group_config is not None,
            "group_label": group_config.group_label if group_config else None,
            "train_only_groups": ", ".join(str(group) for group in group_config.train_only_groups)
                if group_config and group_config.train_only_groups else "none",
            "splits": splits,
            **Util.make_dataset_html_map(state.dataset, dataset_key="original_dataset")
        }

        return html_map
