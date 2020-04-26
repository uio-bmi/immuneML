from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.presentation.PresentationFormat import PresentationFormat
from source.presentation.html.ExploratoryAnalysisHTMLBuilder import ExploratoryAnalysisHTMLBuilder
from source.presentation.html.HPHTMLBuilder import HPHTMLBuilder
from source.presentation.html.SimulationHTMLBuilder import SimulationHTMLBuilder
from source.simulation.SimulationState import SimulationState
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState


class PresentationFactory:

    @staticmethod
    def make_presentation_builder(state, presentation_format: PresentationFormat):
        if isinstance(state, HPOptimizationState) and presentation_format == PresentationFormat.HTML:
            return HPHTMLBuilder
        elif isinstance(state, ExploratoryAnalysisState) and presentation_format == PresentationFormat.HTML:
            return ExploratoryAnalysisHTMLBuilder
        elif isinstance(state, SimulationState) and presentation_format == PresentationFormat.HTML:
            return SimulationHTMLBuilder
        else:
            raise ValueError(f"PresentationFactory: state and format combination ({type(state).__name__}, {presentation_format.name}) "
                             f"is not supported.")
