import os
import logging
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ValidateClusteringInstruction import ValidateClusteringState


class ValidateClusteringHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ValidateClusteringState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = ValidateClusteringHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"ValidateClustering_{state.name}.html"

        TemplateParser.parse(
            template_path=EnvironmentSettings.html_templates_path / "ValidateClustering.html",
            template_map=html_map,
            result_path=result_file
        )

        return result_file

    @staticmethod
    def make_html_map(state: ValidateClusteringState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ValidateClusteringHTMLBuilder.CSS_PATH),
            "name": state.name,
            "immuneML_version": MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "setting_key": state.cl_item.cl_setting.get_key() if state.cl_item and state.cl_item.cl_setting else "N/A",
            "validation_types": ", ".join(state.validation_type) if state.validation_type else "N/A",
            "metrics": ", ".join(state.metrics) if state.metrics else "N/A",
            "show_labels": state.label_config is not None and len(state.label_config.get_labels_by_name()) > 0,
            "labels": [{"name": label} for label in state.label_config.get_labels_by_name()] if state.label_config else [],
            'predictions_table': ValidateClusteringHTMLBuilder._format_predictions_file(state.predictions_path) if state.predictions_path else "N/A",
            'predictions_path': os.path.relpath(state.predictions_path, base_path) if state.predictions_path else "N/A",
            **Util.make_dataset_html_map(state.dataset),
            **ValidateClusteringHTMLBuilder._make_method_based_html_map(state, base_path),
            **ValidateClusteringHTMLBuilder._make_result_based_html_map(state, base_path),
            **ValidateClusteringHTMLBuilder._make_data_reports_html_map(state, base_path)
        }
        return html_map

    @staticmethod
    def _make_data_reports_html_map(state: ValidateClusteringState, base_path: Path) -> dict:
        """Create HTML map entries for data reports."""
        return {
            "data_reports": ValidateClusteringHTMLBuilder._format_reports(
                state.data_report_results, base_path
            )
        }

    @staticmethod
    def _make_method_based_html_map(state: ValidateClusteringState, base_path: Path) -> dict:
        """Create HTML map entries for method-based validation results."""
        html_map = {
            "show_method_based": False,
            "method_based_internal_performance": None,
            "method_based_external_performance": None,
            "method_based_reports": {"has_reports": False}
        }

        if state.method_based_result is not None:
            html_map["show_method_based"] = True

            # Internal performance
            if state.method_based_result.internal_performance:
                html_map["method_based_internal_performance"] = (
                    state.method_based_result.internal_performance.get_df().to_html(
                        border=0, justify='left', max_rows=None, index=False
                    )
                )

            # External performance
            if state.method_based_result.external_performance:
                html_map["method_based_external_performance"] = (
                    state.method_based_result.external_performance.get_df().to_html(
                        border=0, justify='left', max_rows=None, index=False
                    )
                )

            # Reports
            html_map["method_based_reports"] = ValidateClusteringHTMLBuilder._format_reports(
                state.method_based_report_results, base_path
            )

        return html_map

    @staticmethod
    def _make_result_based_html_map(state: ValidateClusteringState, base_path: Path) -> dict:
        """Create HTML map entries for result-based validation results."""
        html_map = {
            "show_result_based": False,
            "result_based_internal_performance": None,
            "result_based_external_performance": None,
            "result_based_reports": {"has_reports": False}
        }

        if state.result_based_result is not None:
            html_map["show_result_based"] = True
            # Internal performance
            if state.result_based_result.internal_performance:
                html_map["result_based_internal_performance"] = (
                    state.result_based_result.internal_performance.get_df().to_html(
                        border=0, justify='left', max_rows=None, index=False
                    )
                )

            # External performance
            if state.result_based_result.external_performance:
                html_map["result_based_external_performance"] = (
                    state.result_based_result.external_performance.get_df().to_html(
                        border=0, justify='left', max_rows=None, index=False
                    )
                )

            # Reports
            html_map["result_based_reports"] = ValidateClusteringHTMLBuilder._format_reports(
                state.result_based_report_results, base_path
            )

        return html_map

    @staticmethod
    def _format_predictions_file(file_path: Path) -> str:
        """Format a predictions CSV file as an HTML table."""
        try:
            df = pd.read_csv(file_path)
            return df.to_html(border=0, classes="prediction-table", max_rows=None, justify='left', index=False)
        except Exception as e:
            logging.warning(f"Error loading predictions: {e}")
            return "Error loading predictions"

    @staticmethod
    def _format_reports(reports: List[ReportResult], base_path: Path) -> dict:
        """Format report results for HTML display."""
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