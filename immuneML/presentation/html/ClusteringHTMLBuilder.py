import operator
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper
from immuneML.workflows.instructions.clustering import ClusteringState

class ClusteringHTMLBuilder:
    """
    A class that will make a HTML file(s) out of ClusteringState object to show what analysis took place in
    the ClsuteringInstruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ClusteringState) -> Path:
        """
        Function that builds the HTML files based on the ClusteringAnalysis state.
        Arguments:
            state: ClusteringAnalysisState object with details and results of the instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
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
            "full_specs": Util.get_full_specs_path(base_path),
            'immuneML_version': MLUtil.get_immuneML_version(),
            "analyses_summary": len(state.clustering_units) > 1 and state.clustering_scores is not None,
            "evaluation_metrics": [metric for metric in list(state.clustering_scores.values())[0]] if state.clustering_scores is not None else None,
            "best_analyses_scores": [{
                "metric": metric,
                "best_score":
                    max({key: x[metric] for key, x in state.clustering_scores.items() if key != "target_score"}.items(), key=operator.itemgetter(1))[0] if state.clustering_scores["target_score"][metric] > 0 else
                    min({key: x[metric] for key, x in state.clustering_scores.items() if key != "target_score"}.items(), key=operator.itemgetter(1))[0] if state.clustering_scores is not None else 0,
            } for metric in list(state.clustering_scores.values())[0] if state.clustering_scores is not None],
            "analyses_scores": [{
                "analyses_name": name,
                "scores": [round(score, 3) for score in analysis.values()]
            } for name, analysis in state.clustering_scores.items()],
            "analyses": [{
                "name": name,
                "dataset_name": analysis.dataset.name if analysis.dataset.name is not None else analysis.dataset.identifier,
                "dataset_type": StringHelper.camel_case_to_word_string(type(analysis.dataset).__name__),
                "example_count": analysis.dataset.get_example_count(),
                "dataset_size": f"{analysis.dataset.get_example_count()} {type(analysis.dataset).__name__.replace('Dataset', 's').lower()}",
                "show_labels": analysis.label_config is not None and len(analysis.label_config.get_labels_by_name()) > 0,
                "labels": [{"name": label.name, "values": str(label.values)[1:-1]}
                           for label in analysis.label_config.get_label_objects()] if analysis.label_config else None,
                "encoding_key": analysis.encoder.name if analysis.encoder is not None else None,
                "encoding_name": StringHelper.camel_case_to_word_string(type(analysis.encoder).__name__) if analysis.encoder is not None
                else None,
                "encoding_params": [{"param_name": key, "param_value": str(value)} for key, value in vars(analysis.encoder).items()] if analysis.encoder is not None else None,
                "show_encoding": analysis.encoder is not None,
                "clustering_key": analysis.clustering_method.name,
                "clustering_name": type(analysis.clustering_method).__name__,
                "clustering_params": [{"param_name": key, "param_value": str(value)} for key, value in analysis.clustering_method.get_params().items()],
                "dimRed_key": analysis.dimensionality_reduction.name if analysis.dimensionality_reduction is not None else None,
                "dimRed_name": type(analysis.dimensionality_reduction).__name__ if analysis.dimensionality_reduction is not None else None,
                "dimRed_params": [{"param_name": key, "param_value": str(value)} for key, value in analysis.dimensionality_reduction.get_params().items()] if analysis.dimensionality_reduction is not None else None,
                "show_dimRed": analysis.dimensionality_reduction is not None,
                "show_evaluation_metrics": state.clustering_scores[name] is not None,
                "evaluation_scores": [{"metric": metric, "score": round(score, 3)} for metric, score in state.clustering_scores[name].items()] if state.clustering_scores is not None and state.clustering_scores[name] is not None else None,
                "report": Util.to_dict_recursive(Util.update_report_paths(analysis.report_result, base_path), base_path)
            } for name, analysis in state.clustering_units.items()]
        }

        for analysis in html_map["analyses"]:
            analysis["show_tables"] = len(analysis["report"]["output_tables"]) > 0 if "output_tables" in analysis["report"] else False
            analysis["show_text"] = len(analysis["report"]["output_text"]) > 0 if "output_text" in analysis["report"] else False
            analysis["show_info"] = analysis["report"]["info"] is not None and len(analysis["report"]["info"]) > 0 if "info" in analysis["report"] else False

        return html_map
