from immuneML.tool_interface.ToolType import ToolType


class InterfaceController:

    @staticmethod
    def tool_interface_controller(tool_type: ToolType, specs: dict):
        """ Runs the main function of the different tool components based on tool type input
        """

        if tool_type == ToolType.ML_TOOL:
            # MLToolComponent.run_ML_tool_component(specs)
            pass
        elif tool_type == ToolType.DATASET_TOOL:
            # DatasetToolComponent.run(specs)
            pass
        else:
            print(f"Invalid input to InterfaceController.'{tool_type}' is not a valid tool definition")
