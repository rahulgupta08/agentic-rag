class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, description, node):

        self.tools[name] = {
            "description": description,
            "node": node
        }

    def get_tools(self):
        return self.tools

    def get_node(self, tool_name):
        return self.tools.get(tool_name, {}).get("node")

    def format_for_prompt(self):

        lines = []

        for name, tool in self.tools.items():
            lines.append(f"{name} : {tool['description']}")

        return "\n".join(lines)