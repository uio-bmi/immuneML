from immuneML.tool_interface.InterfaceController import InterfaceController
from immuneML.tool_interface.ToolType import ToolType


class ToolParser:
    VALID_TOOL_DEFINITIONS = ["ml_tool", "dataset_tool"]

    @staticmethod
    def parse(workflow_specification: dict):
        if "tools" not in workflow_specification:
            return

        specs = workflow_specification["tools"]

        if not ToolParser._check_for_invalid_tool_definitions(specs):
            return

        if "ml_tool" in specs:
            ml_specs = specs.get("ml_tool")
            InterfaceController.tool_interface_controller(ToolType.ML_TOOL, ml_specs)

        if "dataset_tool" in specs:
            dataset_specs = specs.get("dataset_tool")
            InterfaceController.tool_interface_controller(ToolType.DATASET_TOOL, dataset_specs)

    @staticmethod
    def _check_for_invalid_tool_definitions(specs) -> bool:
        """ Checks if there is an invalid tool listed in the YAML spec file
        """
        for tool_def in specs:
            if tool_def not in ToolParser.VALID_TOOL_DEFINITIONS:
                print(f"Tool definition not found: {tool_def}")
                return False

        return True

