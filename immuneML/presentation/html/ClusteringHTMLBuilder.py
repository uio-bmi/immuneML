import io
import os
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringState


class ClusteringHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ClusteringState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = ClusteringHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"Clustering_{state.config.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Clustering.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: ClusteringState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "name": state.config.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "splits": ClusteringHTMLBuilder._make_split_maps(state, base_path),
            **Util.make_dataset_html_map(state.config.dataset)
        }
        return html_map

    @staticmethod
    def _make_split_maps(state: ClusteringState, base_path: Path) -> List[dict]:
        """Create split maps for inline display in main page."""
        splits = []
        for split_id in range(state.config.split_config.split_count):
            validation_types = {
                "has_method_based": "method-based" in state.config.validation_type,
                "has_result_based": "result-based" in state.config.validation_type
            }
            
            split_map = {
                "number": split_id + 1,
                "predictions": {
                    "discovery": {
                        "path": os.path.relpath(state.predictions_paths[split_id]['discovery'], base_path),
                        "data": ClusteringHTMLBuilder._format_predictions_file(state.predictions_paths[split_id]['discovery'])
                    }
                },
                "performance": {
                    "internal": {
                        "show": any(is_internal(m) for m in state.config.metrics),
                        "discovery": ClusteringHTMLBuilder.make_internal_performance_table(state, 'discovery', split_id)
                    },
                    "external": {
                        "show": any(is_external(m) for m in state.config.metrics),
                        "discovery": ClusteringHTMLBuilder.make_external_performance_tables(state, 'discovery', split_id)
                    }
                },
                "setting_details": ClusteringHTMLBuilder._make_setting_details(state, split_id, base_path, validation_types),
                **validation_types
            }

            # Add validation performances if needed
            if validation_types["has_method_based"]:
                split_map["predictions"]["method_based"] = {
                    "path": os.path.relpath(state.predictions_paths[split_id]['method_based_validation'], base_path),
                    "data": ClusteringHTMLBuilder._format_predictions_file(
                        state.predictions_paths[split_id]['method_based_validation'])
                }
                split_map["performance"]["internal"]["method_based"] = ClusteringHTMLBuilder.make_internal_performance_table(
                    state, 'method_based_validation', split_id)
                split_map["performance"]["external"]["method_based"] = ClusteringHTMLBuilder.make_external_performance_tables(
                    state, 'method_based_validation', split_id)

            if validation_types["has_result_based"]:
                split_map["predictions"]["result_based"] = {
                    "path": os.path.relpath(state.predictions_paths[split_id]['result_based_validation'], base_path),
                    "data": ClusteringHTMLBuilder._format_predictions_file(
                        state.predictions_paths[split_id]['result_based_validation'])
                }
                split_map["performance"]["internal"]["result_based"] = ClusteringHTMLBuilder.make_internal_performance_table(
                    state, 'result_based_validation', split_id)
                split_map["performance"]["external"]["result_based"] = ClusteringHTMLBuilder.make_external_performance_tables(
                    state, 'result_based_validation', split_id)

            splits.append(split_map)

        return splits

    @staticmethod
    def _format_predictions_file(file_path: Path) -> str:
        """Read and format predictions file for HTML display."""
        try:
            df = pd.read_csv(file_path)
            return df.to_html(classes="prediction-table", max_rows=None, index=False)
        except:
            return "Error loading predictions"

    @staticmethod
    def _make_setting_details(state: ClusteringState, split_id: int, base_path: Path, validation_types: dict) -> List[dict]:
        """Create detailed pages for each clustering setting."""
        details = []
        for setting in state.config.clustering_settings:
            setting_path = base_path / f"split_{split_id + 1}_{setting.get_key()}.html"
            
            setting_map = {
                "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
                "setting_name": setting.get_key(),
                "split_number": split_id + 1,
                "discovery": ClusteringHTMLBuilder._get_setting_results(state, setting, 'discovery', split_id),
                **validation_types
            }

            if validation_types["has_method_based"]:
                setting_map["method_based"] = ClusteringHTMLBuilder._get_setting_results(
                    state, setting, 'method_based_validation', split_id)

            if validation_types["has_result_based"]:
                setting_map["result_based"] = ClusteringHTMLBuilder._get_setting_results(
                    state, setting, 'result_based_validation', split_id)

            TemplateParser.parse(
                template_path=EnvironmentSettings.html_templates_path / "ClusteringSettingDetails.html",
                template_map=setting_map,
                result_path=setting_path
            )

            details.append({
                "name": setting.get_key(),
                "path": os.path.relpath(setting_path, base_path)
            })

        return details

    @staticmethod
    def _get_setting_results(state: ClusteringState, setting, analysis_type: str, split_id: int) -> dict:
        """Get results for a specific setting and analysis type."""
        try:
            cl_item = state.clustering_items[split_id][analysis_type][setting.get_key()]
            return {
                "predictions_path": state.predictions_paths[split_id][analysis_type],
                "internal_performance": cl_item.internal_performance.get_df().to_html(index=False) if cl_item.internal_performance else None,
                "external_performance": cl_item.external_performance.get_df().to_html(index=False) if cl_item.external_performance else None,
                "reports": ClusteringHTMLBuilder._format_reports(
                    state.cl_item_report_results[split_id][analysis_type][setting.get_key()]
                )
            }
        except (KeyError, AttributeError):
            return {
                "predictions_path": None,
                "internal_performance": None,
                "external_performance": None,
                "reports": "No results available"
            }

    @staticmethod
    def _format_predictions(predictions) -> str:
        """Format predictions for HTML display."""
        df = pd.DataFrame({"cluster": predictions})
        return df.to_html(classes="prediction-table", max_rows=10, show_dimensions=True, index=False)

    @staticmethod
    def make_external_performance_tables(state: ClusteringState, analysis_desc: str, split_id: int) -> List[dict]:
        """Create external performance tables for a specific split and analysis type."""
        cl_item_keys = [cs.get_key() for cs in state.config.clustering_settings]
        external_eval = []
        
        if state.config.label_config is not None:
            for label in state.config.label_config.get_labels_by_name():
                performance_table = {
                    metric.replace("_", " "): [
                        f"{state.clustering_items[split_id][analysis_desc][cl_item].external_performance.get_df().set_index(['metric']).loc[metric, label].item():.3f}"  # Format to 3 decimal places
                        for cl_item in cl_item_keys]
                    for metric in state.config.metrics if is_external(metric)
                }
                s = io.StringIO()
                performance_table = (pd.DataFrame(performance_table, index=cl_item_keys).reset_index()
                                    .rename(columns={'index': 'clustering setting'}))
                performance_table.to_csv(s, sep="\t", index=False)
                external_eval.append({
                    'label': label,
                    'performance_table': Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")
                })
        return external_eval

    @staticmethod
    def make_internal_performance_table(state: ClusteringState, analysis_desc: str, split_id: int) -> str:
        """Create internal performance table for a specific split and analysis type."""
        cl_item_keys = [cs.get_key() for cs in state.config.clustering_settings]
        performance_metric = {
            metric.replace("_", " "): [
                f"{state.clustering_items[split_id][analysis_desc][cl_item].internal_performance.get_df()[metric].values[0]:.3f}"  # Format to 3 decimal places
                for cl_item in cl_item_keys]
            for metric in state.config.metrics if is_internal(metric)
        }

        s = io.StringIO()
        df = (pd.DataFrame(performance_metric, index=cl_item_keys).reset_index()
              .rename(columns={'index': 'clustering setting'}))
        df.to_csv(s, sep="\t", index=False)
        return Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")

    @staticmethod
    def _move_reports_recursive(obj: List[ReportResult], path: Path):
        if isinstance(obj, list) and all(isinstance(item, ReportResult) for item in obj):
            new_obj = []
            for report_result in obj:
                rep_result = Util.update_report_paths(report_result, path)
                for attribute in vars(rep_result):
                    attribute_value = getattr(report_result, attribute)
                    if isinstance(attribute_value, list):
                        for output in attribute_value:
                            if isinstance(output, ReportOutput):
                                output.path = output.path.relative_to(path)
                rep_result = Util.to_dict_recursive(rep_result, path)
                new_obj.append(rep_result)
            return new_obj
        else:
            return obj

    @staticmethod
    def _format_reports(reports) -> dict:
        """Format reports for HTML display with proper handling of plots and other outputs."""
        if isinstance(reports, dict) and 'encoding' in reports:
            reports = reports['encoding']
            
        if isinstance(reports, list) and len(reports) > 0:
            result = {
                "has_reports": True,
                "reports": []
            }
            
            for report in reports:
                if isinstance(report, dict):
                    formatted_report = {
                        "name": report.get('name', 'Report'),
                        "info": report.get('info', ''),
                        "output_figures": [],
                        "output_tables": [],
                        "output_text": []
                    }
                    
                    # Handle output figures
                    for output in report.get('output_figures', []):
                        formatted_report["output_figures"].append({
                            "path": output.get('path', ''),
                            "name": output.get('name', ''),
                            "is_embed": str(output.get('path', '')).endswith(('.html', '.svg'))
                        })
                    
                    # Handle output tables
                    for output in report.get('output_tables', []):
                        formatted_report["output_tables"].append({
                            "path": output.get('path', ''),
                            "name": output.get('name', '')
                        })
                    
                    # Handle output text
                    for output in report.get('output_text', []):
                        formatted_report["output_text"].append({
                            "path": output.get('path', ''),
                            "name": output.get('name', '')
                        })
                    
                    result["reports"].append(formatted_report)
            
            return result
        return {"has_reports": False}
