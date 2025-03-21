import os
from pathlib import Path
from typing import List
import logging

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState


class ClusteringHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ClusteringState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = ClusteringHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"Clustering_{state.config.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Clustering.html",
                             template_map=html_map, result_path=result_file)

        # Generate split pages
        for split_id in range(state.config.split_config.split_count):
            ClusteringHTMLBuilder._make_split_page(state, split_id, base_path)

        # Generate detail pages for each setting in each split
        for split_id in range(state.config.split_config.split_count):
            for setting in state.config.clustering_settings:
                ClusteringHTMLBuilder._make_setting_details_page(state, split_id, setting, base_path)

        return result_file

    @staticmethod
    def make_html_map(state: ClusteringState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "name": state.config.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "clustering_reports": ClusteringHTMLBuilder._format_reports(state.clustering_report_results, base_path),
            "splits": [{"number": i + 1, "path": f"split_{i + 1}.html"} for i in range(state.config.split_config.split_count)],
            **Util.make_dataset_html_map(state.config.dataset)
        }
        return html_map

    @staticmethod
    def _make_split_page(state: ClusteringState, split_id: int, base_path: Path):
        split_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "name": state.config.name,
            "split_number": split_id + 1,
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
                    "discovery": ClusteringHTMLBuilder._make_internal_performance_table(state, 'discovery', split_id)
                },
                "external": {
                    "show": any(is_external(m) for m in state.config.metrics),
                    "discovery": ClusteringHTMLBuilder._make_external_performance_tables(state, 'discovery', split_id)
                }
            },
            "setting_details": ClusteringHTMLBuilder._make_setting_links(state, split_id, base_path),
            "has_method_based": "method_based" in state.config.validation_type,
            "has_result_based": "result_based" in state.config.validation_type,
            "main_page_link": f"Clustering_{state.config.name}.html"
        }

        # Add validation results if present
        if "method_based" in state.config.validation_type:
            split_map["predictions"]["method_based"] = {
                "path": os.path.relpath(state.predictions_paths[split_id]['method_based_validation'], base_path),
                "data": ClusteringHTMLBuilder._format_predictions_file(
                    state.predictions_paths[split_id]['method_based_validation'])
            }
            split_map["performance"]["internal"]["method_based"] = ClusteringHTMLBuilder._make_internal_performance_table(
                state, 'method_based_validation', split_id)
            split_map["performance"]["external"]["method_based"] = ClusteringHTMLBuilder._make_external_performance_tables(
                state, 'method_based_validation', split_id)

        if "result_based" in state.config.validation_type:
            split_map["predictions"]["result_based"] = {
                "path": os.path.relpath(state.predictions_paths[split_id]['result_based_validation'], base_path),
                "data": ClusteringHTMLBuilder._format_predictions_file(
                    state.predictions_paths[split_id]['result_based_validation'])
            }
            split_map["performance"]["internal"]["result_based"] = ClusteringHTMLBuilder._make_internal_performance_table(
                state, 'result_based_validation', split_id)
            split_map["performance"]["external"]["result_based"] = ClusteringHTMLBuilder._make_external_performance_tables(
                state, 'result_based_validation', split_id)

        result_path = base_path / f"split_{split_id + 1}.html"
        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "ClusteringSplit.html",
                             template_map=split_map, result_path=result_path)

    @staticmethod
    def _make_setting_links(state: ClusteringState, split_id: int, base_path: Path) -> List[dict]:
        return [{
            "name": setting.get_key(),
            "path": f"split_{split_id + 1}_{setting.get_key()}.html"
        } for setting in state.config.clustering_settings]

    @staticmethod
    def _make_setting_details_page(state: ClusteringState, split_id: int, setting, base_path: Path):
        template_map = {"css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH), "split_number": split_id + 1,
                        "setting_name": setting.get_key(),
                        "has_method_based": "method_based" in state.config.validation_type,
                        "has_result_based": "result_based" in state.config.validation_type,
                        "discovery": ClusteringHTMLBuilder._get_analysis_results(
                            state, split_id, setting, "discovery", base_path)}

        # Add validation results if present
        if "method_based" in state.config.validation_type:
            template_map["method_based"] = ClusteringHTMLBuilder._get_analysis_results(
                state, split_id, setting, "method_based_validation", base_path)

        if "result_based" in state.config.validation_type:
            template_map["result_based"] = ClusteringHTMLBuilder._get_analysis_results(
                state, split_id, setting, "result_based_validation", base_path)

        result_path = base_path / f"split_{split_id + 1}_{setting.get_key()}.html"
        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "ClusteringSettingDetails.html",
                             template_map=template_map, result_path=result_path)

    @staticmethod
    def _get_analysis_results(state: ClusteringState, split_id: int, setting, analysis_type: str,
                              base_path: Path) -> dict:
        cl_result = state.clustering_items[split_id]
        if hasattr(cl_result, analysis_type):
            analysis_result = getattr(cl_result, analysis_type)
            if analysis_result and setting.get_key() in analysis_result.items:
                item_result = analysis_result.items[setting.get_key()]
                return {
                    "predictions_path": os.path.relpath(state.predictions_paths[split_id][analysis_type], base_path),
                    "internal_performance": item_result.item.internal_performance.get_df().to_html(border=0,
                        justify='left', max_rows=None,
                        index=False) if item_result.item.internal_performance else None,
                    "external_performance": item_result.item.external_performance.get_df().to_html(border=0,
                        justify='left', max_rows=None,
                        index=False) if item_result.item.external_performance else None,
                    "reports": ClusteringHTMLBuilder._format_reports(item_result.report_results, base_path)
                }
        return None

    @staticmethod
    def _format_predictions_file(file_path: Path) -> str:
        try:
            df = pd.read_csv(file_path)
            return df.to_html(border=0, classes="prediction-table", max_rows=None, justify='left', index=False)
        except:
            return "Error loading predictions"

    @staticmethod
    def _format_reports(reports: List[ReportResult], base_path: Path) -> dict:
        if not reports:
            return {"has_reports": False}

        formatted_reports = []
        for report in reports:
            if isinstance(report, ReportResult):
                formatted_report = {
                    "name": report.name,
                    "info": report.info if hasattr(report, "info") else None,
                    "output_figures": [],
                    "output_tables": [],
                    "output_text": []
                }

                # Process figures
                if hasattr(report, "output_figures"):
                    formatted_report["output_figures"] = [{
                        "name": fig.name,
                        "path": os.path.relpath(fig.path, base_path),
                        "is_embed": str(fig.path).endswith(('.html', '.svg'))
                    } for fig in report.output_figures]

                # Process tables
                if hasattr(report, "output_tables"):
                    for table in report.output_tables:
                        try:
                            df = pd.read_csv(table.path)
                            formatted_report["output_tables"].append({
                                "name": table.name,
                                "download_link": os.path.relpath(table.path, base_path),
                                "file_name": os.path.basename(table.path),
                                "table": df.to_html(border=0, justify='left', max_rows=None, index=False)
                            })
                        except Exception as e:
                            logging.warning(f"Error processing table {table.name}: {e}")

                # Process text outputs
                if hasattr(report, "output_text"):
                    formatted_report["output_text"] = [{
                        "name": text.name,
                        "download_link": os.path.relpath(text.path, base_path),
                        "file_name": os.path.basename(text.path),
                        "is_download_link": True
                    } for text in report.output_text]

                formatted_reports.append(formatted_report)

        return {
            "has_reports": True,
            "reports": formatted_reports
        }

    @staticmethod
    def _make_internal_performance_table(state: ClusteringState, analysis_type: str, split_id: int) -> str:
        cl_result = getattr(state.clustering_items[split_id], analysis_type, None)
        if not cl_result:
            return None

        performance_data = {
            "clustering setting": [],
            **{metric: [] for metric in state.config.metrics if is_internal(metric)}
        }

        for setting in state.config.clustering_settings:
            if setting.get_key() in cl_result.items:
                item = cl_result.items[setting.get_key()].item
                performance_data["clustering setting"].append(setting.get_key())
                for metric in state.config.metrics:
                    if is_internal(metric) and item.internal_performance:
                        value = item.internal_performance.get_df()[metric].values[0]
                        performance_data[metric].append(f"{value:.3f}")

        if performance_data["clustering setting"]:
            df = pd.DataFrame(performance_data)
            return df.to_html(border=0, justify='left', max_rows=None, index=False)
        return None

    @staticmethod
    def _make_external_performance_tables(state: ClusteringState, analysis_type: str, split_id: int) -> List[dict]:
        if not state.config.label_config:
            return []

        cl_result = getattr(state.clustering_items[split_id], analysis_type, None)
        if not cl_result:
            return []

        tables = []
        for label in state.config.label_config.get_labels_by_name():
            performance_data = {
                "clustering setting": [],
                **{metric: [] for metric in state.config.metrics if is_external(metric)}
            }

            for setting in state.config.clustering_settings:
                if setting.get_key() in cl_result.items:
                    item = cl_result.items[setting.get_key()].item
                    if item.external_performance:
                        performance_data["clustering setting"].append(setting.get_key())
                        for metric in state.config.metrics:
                            if is_external(metric):
                                value = item.external_performance.get_df().set_index(['metric']).loc[
                                    metric, label].item()
                                performance_data[metric].append(f"{value:.3f}")

            if performance_data["clustering setting"]:
                df = pd.DataFrame(performance_data)
                tables.append({
                    "label": label,
                    "performance_table": df.to_html(border=0, justify='left', max_rows=None, index=False)
                })

        return tables
