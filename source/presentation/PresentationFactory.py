from source.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from source.presentation.PresentationFormat import PresentationFormat
from source.presentation.html.DatasetExportHTMLBuilder import DatasetExportHTMLBuilder
from source.presentation.html.ExploratoryAnalysisHTMLBuilder import ExploratoryAnalysisHTMLBuilder
from source.presentation.html.HPHTMLBuilder import HPHTMLBuilder
from source.presentation.html.MLApplicationHTMLBuilder import MLApplicationHTMLBuilder
from source.presentation.html.SimulationHTMLBuilder import SimulationHTMLBuilder
from source.presentation.html.SubsamplingHTMLBuilder import SubsamplingHTMLBuilder
from source.simulation.SimulationState import SimulationState
from source.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from source.workflows.instructions.ml_model_application.MLApplicationState import MLApplicationState
from source.workflows.instructions.subsampling.SubsamplingState import SubsamplingState


class PresentationFactory:

    @staticmethod
    def make_presentation_builder(state, presentation_format: PresentationFormat):
        if isinstance(state, TrainMLModelState) and presentation_format == PresentationFormat.HTML:
            return HPHTMLBuilder
        elif isinstance(state, ExploratoryAnalysisState) and presentation_format == PresentationFormat.HTML:
            return ExploratoryAnalysisHTMLBuilder
        elif isinstance(state, SimulationState) and presentation_format == PresentationFormat.HTML:
            return SimulationHTMLBuilder
        elif isinstance(state, DatasetExportState) and presentation_format == PresentationFormat.HTML:
            return DatasetExportHTMLBuilder
        elif isinstance(state, MLApplicationState) and presentation_format == PresentationFormat.HTML:
            return MLApplicationHTMLBuilder
        elif isinstance(state, SubsamplingState) and presentation_format == PresentationFormat.HTML:
            return SubsamplingHTMLBuilder
        else:
            raise ValueError(f"PresentationFactory: state and format combination ({type(state).__name__}, {presentation_format.name}) "
                             f"is not supported.")
