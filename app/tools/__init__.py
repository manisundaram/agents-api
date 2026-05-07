"""Agent tools for interacting with external services and local capabilities."""

from .calculator import CalculatorTool
from .file_reader import FileReaderTool 
from .rag_tool import RagTool
from .search_tool import SearchTool

__all__ = [
    "CalculatorTool",
    "FileReaderTool", 
    "RagTool",
    "SearchTool",
]

# Tool registry for easy lookup
AVAILABLE_TOOLS = {
    "calculator": CalculatorTool,
    "file_reader": FileReaderTool,
    "rag": RagTool,
    "search": SearchTool,
}

def get_tool_by_name(tool_name: str):
    """Get a tool instance by name."""
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available tools: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS[tool_name]()