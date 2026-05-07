"""File reader tool for safely reading local files."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from ai_service_kit.logging import Logger

from ..models import ToolCall, ToolResult, ToolCallStatus


class FileReaderTool:
    """Tool for safely reading local files with security restrictions."""
    
    def __init__(self, allowed_directories: list[str] | None = None, max_file_size: int = 10 * 1024 * 1024):
        """
        Initialize the file reader tool.
        
        Args:
            allowed_directories: List of directories that can be read from. If None, uses current working directory.
            max_file_size: Maximum file size in bytes (default: 10MB)
        """
        self.allowed_directories = allowed_directories or [os.getcwd()]
        self.max_file_size = max_file_size
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
    
    @property 
    def name(self) -> str:
        return "file_reader"
    
    @property
    def description(self) -> str:
        return "Read the contents of local files safely within allowed directories"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding to use",
                    "default": "utf-8",
                    "enum": ["utf-8", "ascii", "latin-1", "cp1252"]
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "minimum": 1,
                    "maximum": 10000
                },
                "line_start": {
                    "type": "integer",
                    "description": "Start reading from this line number (1-indexed)",
                    "minimum": 1,
                    "default": 1
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the file reader tool."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            file_path = kwargs.get("file_path")
            if not file_path:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Missing required parameter: file_path"
                )
            
            encoding = kwargs.get("encoding", "utf-8")
            max_lines = kwargs.get("max_lines")
            line_start = kwargs.get("line_start", 1)
            
            Logger.debug(f"Executing file reader tool with path: {file_path}")
            
            # Resolve the file path
            file_path = Path(file_path).resolve()
            
            # Security check: ensure file is within allowed directories
            if not self._is_path_allowed(file_path):
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Access denied: file path is outside allowed directories"
                )
            
            # Check if file exists
            if not file_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"File not found: {file_path}"
                )
            
            # Check if it's a file (not a directory)
            if not file_path.is_file():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Path is not a file: {file_path}"
                )
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"File too large: {file_size} bytes (max: {self.max_file_size} bytes)"
                )
            
            # Read the file content
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    if max_lines is not None or line_start > 1:
                        # Read specific lines
                        lines = f.readlines()
                        total_lines = len(lines)
                        
                        # Adjust for 1-indexed line numbers
                        start_idx = max(0, line_start - 1)
                        
                        if max_lines is not None:
                            end_idx = start_idx + max_lines
                            selected_lines = lines[start_idx:end_idx]
                        else:
                            selected_lines = lines[start_idx:]
                        
                        content = ''.join(selected_lines)
                        lines_read = len(selected_lines)
                    else:
                        # Read entire file
                        content = f.read()
                        lines_read = len(content.splitlines())
                        total_lines = lines_read
                        
            except UnicodeDecodeError:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Failed to decode file with encoding '{encoding}'. Try a different encoding."
                )
            
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            Logger.debug(f"File reader completed successfully, read {lines_read} lines")
            
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.SUCCESS,
                result={
                    "content": content,
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "lines_read": lines_read,
                    "total_lines": total_lines if 'total_lines' in locals() else lines_read,
                    "encoding": encoding,
                    "line_start": line_start,
                    "truncated": max_lines is not None and lines_read == max_lines
                },
                execution_time_ms=execution_time
            )
            
        except PermissionError:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error("Permission denied reading file")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error="Permission denied: cannot read file",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"Unexpected error in file reader tool: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"File read error: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _is_path_allowed(self, file_path: Path) -> bool:
        """Check if the file path is within allowed directories."""
        file_path = file_path.resolve()
        
        for allowed_path in self.allowed_paths:
            try:
                file_path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
    
    def create_tool_call(self, **kwargs: Any) -> ToolCall:
        """Create a ToolCall object for this execution."""
        return ToolCall(
            tool_name=self.name,
            tool_input=kwargs
        )