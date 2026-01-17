import os
from pathlib import Path
from typing import List
import logging

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
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

        # Generate detail pages for each setting in each split
        for split_id in range(state.config.sample_config.split_count):
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
            "splits": ClusteringHTMLBuilder._make_splits_with_settings(state, base_path),
            "show_labels": state.config.label_config is not None and len(state.config.label_config.get_labels_by_name()) > 0,
            "labels": [{"name": label} for label in state.config.label_config.get_labels_by_name()] if state.config.label_config else [],
            **Util.make_dataset_html_map(state.config.dataset),
            **ClusteringHTMLBuilder._make_best_settings_html_map(state, base_path)
        }
        return html_map

    @staticmethod
    def _make_best_settings_html_map(state: ClusteringState, base_path: Path) -> dict:
        """Create HTML map entries for best settings exports and predictions."""
        html_map = {
            "show_best_settings": False,
            "best_settings": [],
            "show_final_predictions": False,
            "final_predictions_table": None,
            "final_predictions_path": None
        }

        # Add best settings zip files
        if state.best_settings_zip_paths:
            html_map["show_best_settings"] = True
            html_map["best_settings"] = [
                {
                    "setting_key": setting_key,
                    "zip_path": os.path.relpath(setting_data['path'], base_path),
                    "zip_filename": os.path.basename(setting_data['path']),
                    "metrics": ", ".join(setting_data['metrics'])
                }
                for setting_key, setting_data in state.best_settings_zip_paths.items()
            ]

        # Add final predictions preview and download link
        if state.final_predictions_path and state.final_predictions_path.exists():
            html_map["show_final_predictions"] = True
            html_map["final_predictions_path"] = os.path.relpath(state.final_predictions_path, base_path)
            html_map["final_predictions_table"] = ClusteringHTMLBuilder._format_predictions_file(
                state.final_predictions_path
            )

        return html_map

    @staticmethod
    def _make_splits_with_settings(state: ClusteringState, base_path: Path) -> List[dict]:
        """Create list of splits with their associated clustering settings."""
        splits = []
        for split_id in range(state.config.sample_config.split_count):
            split_info = {
                "number": split_id + 1,
                "settings": [{
                    "name": setting.get_key(),
                    "path": f"split_{split_id + 1}_{setting.get_key()}.html"
                } for setting in state.config.clustering_settings]
            }
            splits.append(split_info)
        return splits

    @staticmethod
    def _make_setting_details_page(state: ClusteringState, split_id: int, setting, base_path: Path):
        """Generate a details page for a specific clustering setting in a specific split."""
        setting_key = setting.get_key()

        # Get clustering results for this split and setting
        cl_result = state.clustering_items[split_id] if split_id < len(state.clustering_items) else None
        item_result = cl_result.items.get(setting_key) if cl_result else None

        template_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "split_number": split_id + 1,
            "setting_name": setting_key,
            "main_page_link": f"Clustering_{state.config.name}.html",
            "predictions_path": os.path.relpath(state.predictions_paths[split_id], base_path) if state.predictions_paths else None,
            "predictions_table": ClusteringHTMLBuilder._format_predictions_file(state.predictions_paths[split_id]) if state.predictions_paths else None,
            "internal_performance": None,
            "external_performance": None,
            "reports": {"has_reports": False}
        }

        if item_result:
            # Internal performance
            if item_result.item.internal_performance:
                template_map["internal_performance"] = item_result.item.internal_performance.get_df().to_html(
                    border=0, justify='left', max_rows=None, index=False)

            # External performance
            if item_result.item.external_performance:
                template_map["external_performance"] = item_result.item.external_performance.get_df().to_html(
                    border=0, justify='left', max_rows=None, index=False)

            # Reports
            template_map["reports"] = ClusteringHTMLBuilder._format_reports(item_result.report_results, base_path)

        result_path = base_path / f"split_{split_id + 1}_{setting_key}.html"
        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "ClusteringSettingDetails.html",
                             template_map=template_map, result_path=result_path)

    @staticmethod
    def _format_predictions_file(file_path: Path) -> str:
        try:
            df = pd.read_csv(file_path)
            return df.to_html(border=0, classes="prediction-table", max_rows=20, justify='left', index=False)
        except Exception as e:
            logging.warning(f"Error loading predictions: {e}")
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
                    "show_info": hasattr(report, "info") and report.info is not None and len(report.info) > 0,
                    "output_figures": [],
                    "output_tables": [],
                    "output_text": []
                }

                # Process figures
                if hasattr(report, "output_figures") and report.output_figures:
                    formatted_report["output_figures"] = [{
                        "name": fig.name,
                        "path": os.path.relpath(fig.path, base_path),
                        "is_embed": str(fig.path).endswith(('.html', '.svg'))
                    } for fig in report.output_figures]

                # Process tables
                if hasattr(report, "output_tables") and report.output_tables:
                    for table in report.output_tables:
                        try:
                            formatted_report["output_tables"].append({
                                "name": table.name,
                                "download_link": os.path.relpath(table.path, base_path),
                                "file_name": os.path.basename(table.path)
                            })
                        except Exception as e:
                            logging.warning(f"Error processing table {table.name}: {e}")

                # Process text outputs
                if hasattr(report, "output_text") and report.output_text:
                    formatted_report["output_text"] = [{
                        "name": text.name,
                        "download_link": os.path.relpath(text.path, base_path),
                        "file_name": os.path.basename(text.path)
                    } for text in report.output_text]

                formatted_report["show_tables"] = len(formatted_report["output_tables"]) > 0
                formatted_report["show_text"] = len(formatted_report["output_text"]) > 0

                formatted_reports.append(formatted_report)

        return {
            "has_reports": len(formatted_reports) > 0,
            "reports": formatted_reports
        }