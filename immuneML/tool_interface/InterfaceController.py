from immuneML.tool_interface.ToolTable import ToolTable
from immuneML.tool_interface.ToolType import ToolType
from immuneML.tool_interface.interface_components.MLToolComponent import MLToolComponent

toolTable = ToolTable()


# name works as id
def create_component(tool_type: ToolType, name: str, specs: dict):
    if tool_type == ToolType.ML_TOOL:
        new_component = MLToolComponent(name, specs)
        toolTable.add(name, new_component)
    elif tool_type == ToolType.DATASET_TOOL:
        # new_component = DatasetToolComponent(name, specs)
        # tools[name] = new_component
        pass


def run_func(name: str, func: str):
    # TODO: check if tool is running, and start process if not
    # TODO: call function in InterfaceComponent
    a = toolTable
    tool = toolTable.get(name)
    getattr(tool, func)("haha")
    print(tool.tool_path)

    pass


def check_running(name: str):
    # check if component has process running
    tool = toolTable.get(name)
    print(tool.tool_path)


def stop_tool(name: str):
    # stop process
    # TODO: run stop subprocess and close connection in component

    pass
