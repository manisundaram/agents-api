"""Calculator tool for safe mathematical evaluation."""

from __future__ import annotations

import ast
import operator
import asyncio
from typing import Any

from ai_service_kit.logging import Logger

from ..models import ToolCall, ToolResult, ToolCallStatus


class CalculatorTool:
    """Tool for safe mathematical calculations."""
    
    # Supported operators for safe evaluation
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
    }
    
    def __init__(self):
        pass
    
    @property 
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform safe mathematical calculations and evaluations"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'max(1, 2, 3)')"
                },
                "precision": {
                    "type": "integer",
                    "description": "Number of decimal places for rounding the result",
                    "default": 6,
                    "minimum": 0,
                    "maximum": 15
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the calculator tool."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            expression = kwargs.get("expression")
            if not expression:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Missing required parameter: expression"
                )
            
            precision = kwargs.get("precision", 6)
            
            Logger.debug(f"Executing calculator tool with expression: {expression}")
            
            # Clean and prepare expression
            expression = expression.strip()
            if not expression:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Empty expression provided"
                )
            
            # Handle common math functions by replacing them with Python equivalents
            expression = self._preprocess_expression(expression)
            
            # Safely evaluate the expression
            try:
                result = self._safe_eval(expression)
            except SyntaxError:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Invalid mathematical expression: {expression}"
                )
            except ZeroDivisionError:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Division by zero"
                )
            except ValueError as e:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Math error: {str(e)}"
                )
            
            # Round result if it's a float
            if isinstance(result, float) and precision >= 0:
                result = round(result, precision)
            
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            Logger.debug(f"Calculator completed successfully, result: {result}")
            
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.SUCCESS,
                result={
                    "expression": kwargs.get("expression"),  # Original expression
                    "processed_expression": expression,
                    "result": result,
                    "result_type": type(result).__name__
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"Unexpected error in calculator tool: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"Calculation error: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _preprocess_expression(self, expression: str) -> str:
        """Preprocess expression to handle common math functions."""
        import math
        
        # Add math functions to the safe evaluation context
        self.FUNCTIONS.update({
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
            'ceil': math.ceil,
            'floor': math.floor,
        })
        
        # Replace common math constants and functions
        replacements = {
            'π': str(math.pi),
            'pi': str(math.pi),
            'e': str(math.e),
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression
    
    def _safe_eval(self, expression: str) -> Any:
        """Safely evaluate a mathematical expression using AST."""
        try:
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self.FUNCTIONS[func_name](*args)
        elif isinstance(node, ast.Name):
            # Allow access to predefined constants
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            raise ValueError(f"Undefined variable: {node.id}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    def create_tool_call(self, **kwargs: Any) -> ToolCall:
        """Create a ToolCall object for this execution."""
        return ToolCall(
            tool_name=self.name,
            tool_input=kwargs
        )