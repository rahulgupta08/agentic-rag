class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, description, node,capabilities=None):

        self.tools[name] = {
            "name" : name,
            "description": description,
            "node": node,
            "capabilities": capabilities or []
        }

    def get_tools(self):
        return self.tools

    def get_node(self, tool_name):
        return self.tools.get(tool_name, {}).get("node")
    
    def has_capability(self, tool_name, capability):
        tool = self.tools.get(tool_name)
        if not tool:
            return False
        return capability in tool.get("capabilities", [])

    def format_for_prompt(self):

        lines = []

        for name, tool in self.tools.items():
            lines.append(f"{name} : {tool['description']}")

        return "\n".join(lines)