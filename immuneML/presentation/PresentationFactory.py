from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.presentation.PresentationFormat import PresentationFormat
from immuneML.presentation.html.ClusteringHTMLBuilder import ClusteringHTMLBuilder
from immuneML.presentation.html.DatasetExportHTMLBuilder import DatasetExportHTMLBuilder
from immuneML.presentation.html.ExploratoryAnalysisHTMLBuilder import ExploratoryAnalysisHTMLBuilder
from immuneML.presentation.html.FeasibilitySummaryHTMLBuilder import FeasibilitySummaryHTMLBuilder
from immuneML.presentation.html.HPHTMLBuilder import HPHTMLBuilder
from immuneML.presentation.html.LIgOSimulationHTMLBuilder import LIgOSimulationHTMLBuilder
from immuneML.presentation.html.MLApplicationHTMLBuilder import MLApplicationHTMLBuilder
from immuneML.presentation.html.SubsamplingHTMLBuilder import SubsamplingHTMLBuilder
from immuneML.presentation.html.GenModelHTMLBuilder import GenModelHTMLBuilder
from immuneML.simulation.LigoSimState import LigoSimState
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringState
from immuneML.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from immuneML.workflows.instructions.ligo_sim_feasibility.FeasibilitySummaryInstruction import FeasibilitySummaryState
from immuneML.workflows.instructions.ml_model_application.MLApplicationState import MLApplicationState
from immuneML.workflows.instructions.subsampling.SubsamplingState import SubsamplingState
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import GenModelState


class PresentationFactory:

    @staticmethod
    def make_presentation_builder(state, presentation_format: PresentationFormat):
        if isinstance(state, TrainMLModelState) and presentation_format == PresentationFormat.HTML:
            return HPHTMLBuilder
        elif isinstance(state, ExploratoryAnalysisState) and presentation_format == PresentationFormat.HTML:
            return ExploratoryAnalysisHTMLBuilder
        elif isinstance(state, DatasetExportState) and presentation_format == PresentationFormat.HTML:
            return DatasetExportHTMLBuilder
        elif isinstance(state, MLApplicationState) and presentation_format == PresentationFormat.HTML:
            return MLApplicationHTMLBuilder
        elif isinstance(state, SubsamplingState) and presentation_format == PresentationFormat.HTML:
            return SubsamplingHTMLBuilder
        elif isinstance(state, LigoSimState) and presentation_format == PresentationFormat.HTML:
            return LIgOSimulationHTMLBuilder
        elif isinstance(state, FeasibilitySummaryState) and presentation_format == PresentationFormat.HTML:
            return FeasibilitySummaryHTMLBuilder
        elif isinstance(state, GenModelState) and presentation_format == PresentationFormat.HTML:
            return GenModelHTMLBuilder
        elif isinstance(state, ClusteringState) and presentation_format == PresentationFormat.HTML:
            return ClusteringHTMLBuilder
        else:
            raise ValueError(f"PresentationFactory: state and format combination ({type(state).__name__}, {presentation_format.name}) "
                             f"is not supported.")
