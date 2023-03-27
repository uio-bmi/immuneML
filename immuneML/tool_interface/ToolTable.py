from immuneML.tool_interface.interface_components.InterfaceComponent import InterfaceComponent


class ToolTable:
    def __init__(self):
        self.items = {}

    def add(self, name: str, item: InterfaceComponent):
        self.items[name] = item

    def get(self, name: str) -> InterfaceComponent:
        return self.items[name]
