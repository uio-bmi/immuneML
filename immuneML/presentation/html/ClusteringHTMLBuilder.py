import io
import os
from itertools import chain
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.ml_methods.util.Util import Util as MLUtil
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
        result_file = base_path / f"Clustering_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Clustering.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: ClusteringState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "discovery_predictions_path": Path(os.path.relpath(path=str(state.predictions_paths['discovery']), start=str(base_path))),
            "validation_predictions_path": Path(
                os.path.relpath(path=str(state.predictions_paths['validation']), start=str(base_path))),
            "performance_table_internal_val": ClusteringHTMLBuilder.make_internal_performance_table(state, 'validation'),
            "performance_table_internal_disc": ClusteringHTMLBuilder.make_internal_performance_table(state, 'discovery'),
            "label_eval_tables_val": ClusteringHTMLBuilder.make_external_performance_tables(state, 'validation'),
            "label_eval_tables_disc": ClusteringHTMLBuilder.make_external_performance_tables(state, 'discovery'),
            "cluster_setting_pages_val": ClusteringHTMLBuilder.make_cluster_setting_pages(state, base_path, 'validation'),
            "cluster_setting_pages_disc": ClusteringHTMLBuilder.make_cluster_setting_pages(state, base_path, 'discovery')
        }

        html_map = {**html_map, **{'show_internal_eval': sum([is_internal(m) for m in state.metrics]) > 0,
                                   'show_external_eval': sum([is_external(m) for m in state.metrics]) > 0,
                                   "show_cluster_setting_pages_val": len(html_map['cluster_setting_pages_val']) > 0,
                                   "show_cluster_setting_pages_disc": len(html_map['cluster_setting_pages_disc']) > 0},
                    **Util.make_dataset_html_map(state.dataset)}

        return html_map

    @staticmethod
    def make_cluster_setting_pages(state: ClusteringState, base_path: Path, analysis_desc: str) -> List[dict]:
        cluster_setting_pages = []
        for cl_setting in state.clustering_settings:
            if (state.cl_item_report_results[analysis_desc][cl_setting.get_key()] is not None
                    and len(state.cl_item_report_results[analysis_desc][cl_setting.get_key()]) > 0):
                cluster_setting_pages.append({
                    'name': cl_setting.get_key(),
                    'path': ClusteringHTMLBuilder.make_cluster_setting_page(state, cl_setting, base_path, analysis_desc)
                })
        return cluster_setting_pages

    @staticmethod
    def make_cluster_setting_page(state: ClusteringState, cl_setting, base_path: Path, analysis_desc: str) -> Path:
        result_path = base_path / f"{state.name}_{analysis_desc}_clustering_setting_{cl_setting.get_key()}.html"

        if state.cl_item_report_results[analysis_desc][cl_setting.get_key()]:
            report_results = Util.to_dict_recursive(state.cl_item_report_results[analysis_desc][cl_setting.get_key()], base_path)
            report_results = list(chain.from_iterable(report_results.values()))
            report_results = ClusteringHTMLBuilder._move_reports_recursive(report_results, base_path)
        else:
            report_results = None

        html_map = {
            "css_style": Util.get_css_content(ClusteringHTMLBuilder.CSS_PATH),
            "cl_setting_name": cl_setting.get_key(),
            "analysis_desc": analysis_desc,
            "reports": report_results,
            'show_internal_performances': any(is_internal(m) for m in state.metrics)
                                          and state.clustering_items[analysis_desc][cl_setting.get_key()].internal_performance,
            'show_external_performances': any(is_external(m) for m in state.metrics)
                                          and state.clustering_items[analysis_desc][cl_setting.get_key()].external_performance
        }

        if state.clustering_items[analysis_desc][cl_setting.get_key()].internal_performance.path is not None and \
                state.clustering_items[analysis_desc][cl_setting.get_key()].internal_performance.path.is_file():
            with state.clustering_items[analysis_desc][cl_setting.get_key()].internal_performance.path.open('r') as file:
                html_map["internal_performances"] = Util.get_table_string_from_csv_string(file.read(), separator=",")
        else:
            html_map["show_internal_performances"] = False

        if state.clustering_items[analysis_desc][cl_setting.get_key()].external_performance.path is not None and \
                state.clustering_items[analysis_desc][cl_setting.get_key()].external_performance.path.is_file():

            with state.clustering_items[analysis_desc][cl_setting.get_key()].external_performance.path.open('r') as file:
                html_map['external_performances'] = Util.get_table_string_from_csv_string(csv_string=file.read(), separator=",", has_header=True)
        else:
            html_map["external_performances"] = False

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "ClusteringSettingDetails.html",
                             template_map=html_map, result_path=result_path)

        return result_path.relative_to(base_path)

    @staticmethod
    def make_external_performance_tables(state: ClusteringState, analysis_desc: str) -> List[dict]:
        cl_item_keys = [cs.get_key() for cs in state.clustering_settings]
        external_eval = []
        for label in state.label_config.get_labels_by_name():
            performance_table = {
                metric.replace("_", " "): [
                    state.clustering_items[analysis_desc][cl_item].external_performance.get_df().set_index(['metric']).loc[metric, label].item()
                    for cl_item in cl_item_keys]
                for metric in state.metrics if is_external(metric)
            }
            s = io.StringIO()
            performance_table = (pd.DataFrame(performance_table, index=cl_item_keys).reset_index()
                                 .rename(columns={'index': 'cluster setting'}))
            performance_table.to_csv(s, sep="\t", index=False)
            external_eval.append({
                'label': label,
                'performance_table': Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")
            })
        return external_eval

    @staticmethod
    def make_internal_performance_table(state: ClusteringState, analysis_desc: str) -> str:
        cl_item_keys = [cs.get_key() for cs in state.clustering_settings]
        performance_metric = {
            metric.replace("_", " "): [state.clustering_items[analysis_desc][cl_item].internal_performance.get_df()[metric].values[0]
                                       for cl_item in cl_item_keys]
            for metric in state.metrics if is_internal(metric)
        }

        s = io.StringIO()
        df = (pd.DataFrame(performance_metric, index=cl_item_keys).reset_index()
              .rename(columns={'index': 'cluster setting'}))
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
