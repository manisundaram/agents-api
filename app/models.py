"""Pydantic models for agents-api requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallStatus(str, Enum):
    """Status of a tool call execution."""
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ReasoningStep(BaseModel):
    """A step in the agent's reasoning process."""
    step: int = Field(..., description="Step number in the reasoning process")
    thought: str = Field(..., description="The agent's reasoning at this step")
    action: str | None = Field(None, description="Action to take (if any)")
    action_input: dict[str, Any] | None = Field(None, description="Input for the action")
    observation: str | None = Field(None, description="Result of the action")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this step occurred")


class ToolCall(BaseModel):
    """A call to an agent tool."""
    tool_name: str = Field(..., description="Name of the tool being called")
    tool_input: dict[str, Any] = Field(..., description="Input parameters for the tool")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the tool was called")
    
    
class ToolResult(BaseModel):
    """Result from executing a tool."""
    tool_name: str = Field(..., description="Name of the tool that was executed")
    status: ToolCallStatus = Field(..., description="Status of the tool execution")
    result: str | dict[str, Any] | None = Field(None, description="Result data from the tool")
    error: str | None = Field(None, description="Error message if execution failed")
    execution_time_ms: int | None = Field(None, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the tool execution completed")


class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: dict[str, Any] = Field(..., description="JSON schema for tool inputs")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="Example usage")


class AgentRequest(BaseModel):
    """Request to execute a single agent reasoning loop."""
    query: str = Field(..., description="The user's query or task")
    tools: list[str] = Field(default_factory=list, description="List of tool names to make available")
    max_steps: int = Field(default=10, ge=1, le=50, description="Maximum reasoning steps")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature for generation")
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="Timeout for the entire execution")
    memory_key: str | None = Field(None, description="Key to retrieve/store conversation memory")


class AgentResponse(BaseModel):
    """Response from a single agent execution."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Final answer from the agent")
    trace: list[ReasoningStep] = Field(..., description="Full reasoning trace")
    tool_calls: list[ToolCall] = Field(..., description="All tool calls made during execution")
    tool_results: list[ToolResult] = Field(..., description="Results from all tool executions")
    total_steps: int = Field(..., description="Number of reasoning steps taken")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    success: bool = Field(..., description="Whether the agent completed successfully")
    error: str | None = Field(None, description="Error message if execution failed")


class MultiAgentStep(BaseModel):
    """A step in multi-agent workflow execution."""
    agent_role: Literal["planner", "worker", "reviewer"] = Field(..., description="Which agent performed this step")
    step: int = Field(..., description="Step number in the multi-agent process")
    input: str = Field(..., description="Input to this agent")
    output: str = Field(..., description="Output from this agent")
    reasoning: list[ReasoningStep] = Field(default_factory=list, description="Reasoning steps for this agent")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls made by this agent")
    tool_results: list[ToolResult] = Field(default_factory=list, description="Tool results for this agent")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this step completed")


class MultiAgentPlan(BaseModel):
    """Plan created by the planner agent."""
    task: str = Field(..., description="The original task")
    subtasks: list[str] = Field(..., description="List of subtasks to execute")
    reasoning: str = Field(..., description="Planner's reasoning for the plan")
    estimated_complexity: Literal["low", "medium", "high"] = Field(..., description="Estimated task complexity")


class MultiAgentRequest(BaseModel):
    """Request to execute a multi-agent workflow."""
    query: str = Field(..., description="The user's query or task")
    tools: list[str] = Field(default_factory=list, description="List of tool names to make available")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Maximum planner-worker-reviewer iterations")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature for generation")
    timeout_seconds: int = Field(default=600, ge=1, le=3600, description="Timeout for the entire workflow")


class MultiAgentResult(BaseModel):
    """Result from multi-agent workflow execution."""
    query: str = Field(..., description="The original query")
    plan: MultiAgentPlan = Field(..., description="The execution plan created by the planner")
    steps: list[MultiAgentStep] = Field(..., description="All steps in the workflow")
    final_answer: str = Field(..., description="Final answer after review")
    iterations: int = Field(..., description="Number of planner-worker-reviewer iterations")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    success: bool = Field(..., description="Whether the workflow completed successfully")
    error: str | None = Field(None, description="Error message if execution failed")


class CacheEntry(BaseModel):
    """Entry in the semantic cache."""
    query: str = Field(..., description="Original query")
    embedding: list[float] = Field(..., description="Query embedding vector")
    response: str = Field(..., description="Cached response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When the entry was cached")
    access_count: int = Field(default=1, description="Number of times this entry was accessed")


class GuardRailResult(BaseModel):
    """Result from guardrail validation."""
    passed: bool = Field(..., description="Whether validation passed")
    rule_name: str = Field(..., description="Name of the guardrail rule")
    message: str | None = Field(None, description="Validation message or error")
    filtered_content: str | None = Field(None, description="Content after filtering (if applicable)")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in the validation result")


class MemoryEntry(BaseModel):
    """Entry in agent memory."""
    key: str = Field(..., description="Memory key identifier")
    content: str = Field(..., description="Memory content")
    memory_type: Literal["short_term", "long_term"] = Field(..., description="Type of memory")
    created_at: datetime = Field(default_factory=datetime.now, description="When the memory was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When the memory was last updated")
    access_count: int = Field(default=0, description="Number of times accessed")


class RouteDecision(BaseModel):
    """Decision made by the model router."""
    selected_provider: str = Field(..., description="Selected provider name")
    selected_model: str = Field(..., description="Selected model name")
    reasoning: str = Field(..., description="Reasoning for the routing decision")
    fallback_used: bool = Field(default=False, description="Whether a fallback was used")
    route_type: Literal["cheap", "expensive", "fallback"] = Field(..., description="Type of routing decision")


# Response models for API endpoints
class ToolsResponse(BaseModel):
    """Response listing available tools."""
    tools: list[ToolInfo] = Field(..., description="List of available tools")
    total_count: int = Field(..., description="Total number of available tools")


class TraceResponse(BaseModel):
    """Response for agent trace debug endpoint."""
    last_execution: AgentResponse | MultiAgentResult | None = Field(None, description="Last agent execution result")
    cache_stats: dict[str, Any] = Field(..., description="Semantic cache statistics")
    memory_stats: dict[str, Any] = Field(..., description="Memory usage statistics")
    tool_stats: dict[str, int] = Field(..., description="Tool usage statistics")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code for programmatic handling")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")