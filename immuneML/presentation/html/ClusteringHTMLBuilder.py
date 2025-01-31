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
    def _format_output_items(output_list: List[ReportOutput]) -> List[dict]:
        """Helper method to format report outputs (figures, tables, text)"""
        return [{
            "path": str(output.path),
            "name": output.name,
            "is_embed": str(output.path).endswith(('.html', '.svg'))
        } for output in output_list]

    @staticmethod
    def _format_reports(reports, base_path: Path) -> dict:
        """Format reports for HTML display and copy files to HTML output directory."""
        if isinstance(reports, dict) and 'encoding' in reports:
            reports = reports['encoding']

        if not (isinstance(reports, list) and reports):
            return {"has_reports": False}

        result = {
            "has_reports": True,
            "reports": []
        }

        for report in reports:
            if isinstance(report, ReportResult):
                # Copy report files to HTML dir and get relative paths
                report_copy = Util.update_report_paths(report, base_path)
                
                formatted_report = {
                    "name": report_copy.name,
                    "info": report_copy.info,
                    "output_figures": [],
                    "output_tables": [],
                    "output_text": []
                }

                # Process all outputs
                for output_type in ['output_figures', 'output_tables', 'output_text']:
                    for output in getattr(report_copy, output_type):
                        try:
                            rel_path = os.path.relpath(output.path, base_path)
                            formatted_report[output_type].append({
                                "path": str(rel_path),
                                "name": output.name,
                                "is_embed": str(output.path).endswith(('.html', '.svg'))
                            })
                        except ValueError as e:
                            print(f"Error making path relative: {e}")

                result["reports"].append(formatted_report)

        return result

    @staticmethod
    def _add_validation_results(split_map: dict, state: ClusteringState, split_id: int,
                                validation_type: str, base_path: Path):
        """Helper method to add validation results to split map"""
        split_map["predictions"][validation_type] = {
            "path": os.path.relpath(state.predictions_paths[split_id][f'{validation_type}_validation'], base_path),
            "data": ClusteringHTMLBuilder._format_predictions_file(
                state.predictions_paths[split_id][f'{validation_type}_validation'])
        }
        split_map["performance"]["internal"][validation_type] = \
            ClusteringHTMLBuilder.make_internal_performance_table(state, f'{validation_type}_validation', split_id)
        split_map["performance"]["external"][validation_type] = \
            ClusteringHTMLBuilder.make_external_performance_tables(state, f'{validation_type}_validation', split_id)

    @staticmethod
    def _make_split_maps(state: ClusteringState, base_path: Path) -> List[dict]:
        """Create split maps for inline display in main page."""
        splits = []
        for split_id in range(state.config.split_config.split_count):
            validation_types = {
                "has_method_based": "method_based" in state.config.validation_type,
                "has_result_based": "result_based" in state.config.validation_type
            }

            split_map = {
                "number": split_id + 1,
                "predictions": {
                    "discovery": {
                        "path": os.path.relpath(state.predictions_paths[split_id]['discovery'], base_path),
                        "data": ClusteringHTMLBuilder._format_predictions_file(
                            state.predictions_paths[split_id]['discovery'])
                    }
                },
                "performance": {
                    "internal": {
                        "show": any(is_internal(m) for m in state.config.metrics),
                        "discovery": ClusteringHTMLBuilder.make_internal_performance_table(state, 'discovery', split_id)
                    },
                    "external": {
                        "show": any(is_external(m) for m in state.config.metrics),
                        "discovery": ClusteringHTMLBuilder.make_external_performance_tables(state, 'discovery',
                                                                                            split_id)
                    }
                },
                "setting_details": ClusteringHTMLBuilder._make_setting_details(state, split_id, base_path,
                                                                               validation_types),
                **validation_types
            }

            if validation_types["has_method_based"]:
                ClusteringHTMLBuilder._add_validation_results(split_map, state, split_id, "method_based", base_path)
            if validation_types["has_result_based"]:
                ClusteringHTMLBuilder._add_validation_results(split_map, state, split_id, "result_based", base_path)

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
            # Create setting details HTML file in HTML_output directory
            setting_path = base_path / f"split_{split_id + 1}_{setting.get_key()}.html"

            setting_map = {
                "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
                "setting_name": setting.get_key(),
                "split_number": split_id + 1,
                "discovery": ClusteringHTMLBuilder._get_setting_results(state, setting, 'discovery', split_id, base_path),
                **validation_types
            }

            if validation_types["has_method_based"]:
                setting_map["method_based"] = ClusteringHTMLBuilder._get_setting_results(
                    state, setting, 'method_based_validation', split_id, base_path)

            if validation_types["has_result_based"]:
                setting_map["result_based"] = ClusteringHTMLBuilder._get_setting_results(
                    state, setting, 'result_based_validation', split_id, base_path)

            TemplateParser.parse(
                template_path=EnvironmentSettings.html_templates_path / "ClusteringSettingDetails.html",
                template_map=setting_map,
                result_path=setting_path
            )

            # Add relative path to the setting details file
            details.append({
                "name": setting.get_key(),
                "path": os.path.relpath(setting_path, base_path)
            })

        return details

    @staticmethod
    def _get_setting_results(state: ClusteringState, setting, analysis_type: str, split_id: int, base_path: Path) -> dict:
        """Get results for a specific setting and analysis type."""
        try:
            cl_item = state.clustering_items[split_id][analysis_type][setting.get_key()]
            reports = state.cl_item_report_results[split_id][analysis_type][setting.get_key()]
            
            # Update report paths and convert to dict format
            processed_reports = []
            for report_type in reports.keys():
                for report in reports[report_type]:
                    if isinstance(report, ReportResult):  # Add type check
                        try:
                            report_with_paths = Util.update_report_paths(report, base_path)
                            report_dict = Util.to_dict_recursive(report_with_paths, base_path)
                            processed_reports.append(report_dict)
                        except Exception as e:
                            print(f"Error processing report {report.name}: {e}")
            
            # Format performance tables
            internal_df = cl_item.internal_performance.get_df() if cl_item.internal_performance else None
            external_df = cl_item.external_performance.get_df() if cl_item.external_performance else None
            
            result = {
                "predictions_path": os.path.relpath(state.predictions_paths[split_id][analysis_type], base_path),
                "internal_performance": internal_df.to_html(
                    index=False, classes="table-container", float_format=lambda x: f"{x:.3f}"
                ) if internal_df is not None else None,
                "external_performance": external_df.to_html(
                    index=False, classes="table-container", float_format=lambda x: f"{x:.3f}"
                ) if external_df is not None else None,
                "reports": {
                    "has_reports": bool(processed_reports),
                    "reports": processed_reports
                }
            }
            return result
        except (KeyError, AttributeError) as e:
            print(f"Error getting setting results for {setting.get_key()}, {analysis_type}: {e}")
            return {
                "predictions_path": None,
                "internal_performance": None,
                "external_performance": None,
                "reports": {"has_reports": False}
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
                # Create performance table for this label
                performance_table = {
                    "clustering setting": cl_item_keys,
                    **{
                        metric.replace("_", " "): [
                            f"{state.clustering_items[split_id][analysis_desc][cl_item].external_performance.get_df().set_index(['metric']).loc[metric, label].item():.3f}"
                            for cl_item in cl_item_keys
                        ]
                        for metric in state.config.metrics if is_external(metric)
                    }
                }
                
                # Convert to DataFrame and format
                df = pd.DataFrame(performance_table)
                s = io.StringIO()
                df.to_csv(s, sep="\t", index=False)
                
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
                f"{state.clustering_items[split_id][analysis_desc][cl_item].internal_performance.get_df()[metric].values[0]:.3f}"
                # Format to 3 decimal places
                for cl_item in cl_item_keys]
            for metric in state.config.metrics if is_internal(metric)
        }

        s = io.StringIO()
        df = (pd.DataFrame(performance_metric, index=cl_item_keys).reset_index()
              .rename(columns={'index': 'clustering setting'}))
        df.to_csv(s, sep="\t", index=False)
        return Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")
