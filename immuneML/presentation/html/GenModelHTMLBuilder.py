import itertools
import os
from pathlib import Path

from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelState
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import GenModelState


class GenModelHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: GenModelState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = GenModelHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"GenModel_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "GenModel.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state, base_path: Path) -> dict:
        exported_datasets = [{'name': key,
                              'path': os.path.relpath(path=Util.make_downloadable_zip(state.result_path, path),
                                                      start=base_path)}
                             for key, path in state.exported_datasets.items()]
        html_map = {
            "css_style": Util.get_css_content(GenModelHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "function": "Applied" if isinstance(state, ApplyGenModelState) else "Trained",
            'exported_datasets': exported_datasets,
            "show_exported_datasets": len(exported_datasets) > 0,
        }

        html_map = {**html_map, **{
            'show_reports': any(len(rep_results) > 0 for rep_results in state.report_results.values()),
            'reports': list(itertools.chain.from_iterable(
                [Util.to_dict_recursive(Util.update_report_paths(report_result, base_path), base_path)
                 for report_result in state.report_results[report_type]]
                for report_type in state.report_results.keys())),
        }}

        if hasattr(state, "generated_dataset") and state.generated_dataset is not None:
            if "generated_dataset" in state.exported_datasets.keys():
                html_map = {**html_map,
                            **Util.make_dataset_html_map(state.generated_dataset, "generated_dataset"),
                            **{"show_generated_dataset": True}}

        if hasattr(state, "combined_dataset") and state.combined_dataset is not None:
            if "combined_dataset" in state.exported_datasets.keys():
                html_map = {**html_map,
                            **Util.make_dataset_html_map(state.combined_dataset, "combined_dataset"),
                            **{"show_combined_dataset": True}}

        return html_map
