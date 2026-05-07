"""Single-agent ReAct-style reasoning implementation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ai_service_kit.logging import Logger

from .config import get_settings
from .providers import create_provider
from .models import (
    AgentRequest, AgentResponse, ReasoningStep, ToolCall, ToolResult, ToolCallStatus
)
from .tools import get_tool_by_name, AVAILABLE_TOOLS
from .guardrails import GuardrailsManager
from .memory import MemoryManager


class ReactAgent:
    """Single-agent ReAct (Reasoning and Acting) implementation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.provider = None
        self.guardrails = GuardrailsManager()
        self.memory = MemoryManager()
        
    async def _ensure_provider(self):
        """Ensure the provider is initialized."""
        if self.provider is None:
            self.provider = create_provider(
                provider_type=self.settings.provider_type,
                config=self.settings.provider_config()
            )
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute a single-agent ReAct reasoning loop."""
        start_time = time.time()
        
        Logger.info(f"Starting agent execution for query: {request.query[:100]}...")
        
        try:
            # Validate request through guardrails
            validation_result = await self.guardrails.validate_input(request.query)
            if not validation_result.passed:
                return AgentResponse(
                    query=request.query,
                    answer="I cannot process this request due to content policy restrictions.",
                    trace=[],
                    tool_calls=[],
                    tool_results=[],
                    total_steps=0,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    error=f"Input validation failed: {validation_result.message}"
                )
            
            # Initialize provider
            await self._ensure_provider()
            
            # Load memory context if provided
            memory_context = ""
            if request.memory_key:
                memory_context = await self.memory.get_memory(request.memory_key)
            
            # Execute the reasoning loop
            trace = []
            tool_calls = []
            tool_results = []
            
            current_observation = f"User Query: {request.query}"
            if memory_context:
                current_observation += f"\n\nPrevious Context: {memory_context}"
            
            for step in range(1, request.max_steps + 1):
                Logger.debug(f"Agent reasoning step {step}/{request.max_steps}")
                
                # Generate reasoning step
                reasoning_step = await self._generate_reasoning_step(
                    step=step,
                    query=request.query,
                    observation=current_observation,
                    trace=trace,
                    available_tools=request.tools,
                    temperature=request.temperature
                )
                
                trace.append(reasoning_step)
                
                # Check if agent wants to use a tool
                if reasoning_step.action and reasoning_step.action_input:
                    tool_call, tool_result = await self._execute_tool(
                        reasoning_step.action,
                        reasoning_step.action_input
                    )
                    
                    if tool_call:
                        tool_calls.append(tool_call)
                    if tool_result:
                        tool_results.append(tool_result)
                        reasoning_step.observation = self._format_tool_result(tool_result)
                        current_observation = reasoning_step.observation
                
                # Check for completion signals
                if self._is_complete(reasoning_step.thought, reasoning_step.observation):
                    Logger.debug(f"Agent completed reasoning after {step} steps")
                    break
                    
                # Check timeout
                if time.time() - start_time > request.timeout_seconds:
                    Logger.warning("Agent execution timed out")
                    break
            
            # Generate final answer
            final_answer = await self._generate_final_answer(
                query=request.query,
                trace=trace,
                temperature=request.temperature
            )
            
            # Validate output through guardrails
            output_validation = await self.guardrails.validate_output(final_answer)
            if not output_validation.passed:
                final_answer = output_validation.filtered_content or "I cannot provide a response due to content policy restrictions."
            
            # Save to memory if requested
            if request.memory_key:
                await self.memory.save_memory(
                    key=request.memory_key,
                    content=f"Q: {request.query}\nA: {final_answer}",
                    memory_type="short_term"
                )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            Logger.info(f"Agent execution completed successfully in {execution_time_ms}ms with {len(trace)} steps")
            
            return AgentResponse(
                query=request.query,
                answer=final_answer,
                trace=trace,
                tool_calls=tool_calls,
                tool_results=tool_results,
                total_steps=len(trace),
                execution_time_ms=execution_time_ms,
                success=True
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            Logger.error(f"Agent execution failed: {e}", exc_info=True)
            
            return AgentResponse(
                query=request.query,
                answer="I encountered an error while processing your request.",
                trace=trace if 'trace' in locals() else [],
                tool_calls=tool_calls if 'tool_calls' in locals() else [],
                tool_results=tool_results if 'tool_results' in locals() else [],
                total_steps=len(trace) if 'trace' in locals() else 0,
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
    
    async def _generate_reasoning_step(
        self,
        step: int,
        query: str,
        observation: str,
        trace: list[ReasoningStep],
        available_tools: list[str],
        temperature: float
    ) -> ReasoningStep:
        """Generate a single reasoning step using the LLM."""
        
        # Build context from previous steps
        context = f"User Query: {query}\n\n"
        
        if trace:
            context += "Previous reasoning steps:\n"
            for prev_step in trace:
                context += f"Step {prev_step.step}: {prev_step.thought}\n"
                if prev_step.action:
                    context += f"Action: {prev_step.action}({prev_step.action_input})\n"
                if prev_step.observation:
                    context += f"Observation: {prev_step.observation}\n"
            context += "\n"
        
        context += f"Current observation: {observation}\n\n"
        
        # Add available tools information
        if available_tools:
            tools_info = []
            for tool_name in available_tools:
                if tool_name in AVAILABLE_TOOLS:
                    tool = get_tool_by_name(tool_name)
                    tools_info.append(f"- {tool.name}: {tool.description}")
            
            if tools_info:
                context += "Available tools:\n" + "\n".join(tools_info) + "\n\n"
        
        # ReAct prompt
        prompt = f"""{context}You are an AI assistant that reasons step by step to solve problems.

For this step {step}, think carefully about the current situation and decide what to do next.

You can either:
1. Use a tool by specifying: Action: <tool_name> with ActionInput: <input_parameters>
2. Provide a final answer if you have enough information
3. Continue reasoning to gather more information

Format your response as:
Thought: <your reasoning about what to do next>
Action: <tool_name or "none" if no action needed>
ActionInput: <parameters for the tool as JSON object, or null if no action>

Think step by step and be specific about your reasoning."""
        
        try:
            response = await self.provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=500
            )
            
            # Parse the response
            thought, action, action_input = self._parse_reasoning_response(response)
            
            return ReasoningStep(
                step=step,
                thought=thought,
                action=action,
                action_input=action_input
            )
            
        except Exception as e:
            Logger.error(f"Failed to generate reasoning step: {e}")
            return ReasoningStep(
                step=step,
                thought=f"Error in reasoning step: {str(e)}",
                action=None,
                action_input=None
            )
    
    def _parse_reasoning_response(self, response: str) -> tuple[str, str | None, dict[str, Any] | None]:
        """Parse the LLM response into thought, action, and action input."""
        thought = ""
        action = None
        action_input = None
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Thought:'):
                thought = line[8:].strip()
            elif line.startswith('Action:'):
                action_text = line[7:].strip()
                if action_text.lower() != "none":
                    action = action_text
            elif line.startswith('ActionInput:'):
                input_text = line[12:].strip()
                if input_text.lower() not in ["null", "none", ""]:
                    try:
                        action_input = json.loads(input_text)
                    except json.JSONDecodeError:
                        Logger.warning(f"Failed to parse action input as JSON: {input_text}")
                        # Try to create a simple dict if it's a single value
                        action_input = {"input": input_text}
        
        return thought, action, action_input
    
    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[ToolCall | None, ToolResult | None]:
        """Execute a tool and return the call and result."""
        try:
            if tool_name not in AVAILABLE_TOOLS:
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolCallStatus.ERROR,
                    error=f"Unknown tool: {tool_name}"
                )
                return None, error_result
            
            tool = get_tool_by_name(tool_name)
            tool_call = tool.create_tool_call(**tool_input)
            
            Logger.debug(f"Executing tool: {tool_name} with input: {tool_input}")
            
            # Execute with guardrails wrapper
            result = await self.guardrails.safe_tool_execution(tool, **tool_input)
            
            return tool_call, result
            
        except Exception as e:
            Logger.error(f"Tool execution failed: {e}")
            error_result = ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.ERROR,
                error=f"Tool execution error: {str(e)}"
            )
            return None, error_result
    
    def _format_tool_result(self, tool_result: ToolResult) -> str:
        """Format a tool result for the agent's observation."""
        if tool_result.status == ToolCallStatus.SUCCESS:
            if isinstance(tool_result.result, dict):
                # Extract the most relevant information
                if "formatted_answer" in tool_result.result:
                    return tool_result.result["formatted_answer"]
                elif "formatted_summary" in tool_result.result:
                    return tool_result.result["formatted_summary"]
                elif "result" in tool_result.result:
                    return str(tool_result.result["result"])
                elif "content" in tool_result.result:
                    content = tool_result.result["content"]
                    # Truncate very long content
                    if len(content) > 2000:
                        content = content[:2000] + "... (truncated)"
                    return content
                else:
                    return str(tool_result.result)
            else:
                return str(tool_result.result)
        else:
            return f"Tool error: {tool_result.error}"
    
    def _is_complete(self, thought: str, observation: str | None) -> bool:
        """Check if the agent has completed its reasoning."""
        completion_signals = [
            "final answer",
            "conclusion",
            "complete",
            "finished",
            "done",
            "answer is"
        ]
        
        text_to_check = (thought + " " + (observation or "")).lower()
        return any(signal in text_to_check for signal in completion_signals)
    
    async def _generate_final_answer(
        self,
        query: str,
        trace: list[ReasoningStep],
        temperature: float
    ) -> str:
        """Generate the final answer based on the reasoning trace."""
        
        context = f"Original query: {query}\n\n"
        context += "Reasoning process:\n"
        
        for step in trace:
            context += f"Step {step.step}: {step.thought}\n"
            if step.action and step.observation:
                context += f"Tool used: {step.action}\n"
                context += f"Result: {step.observation}\n"
            context += "\n"
        
        prompt = f"""{context}Based on the reasoning process above, provide a clear, concise, and helpful final answer to the user's query.

The answer should:
1. Directly address the user's question
2. Be based on the information gathered during reasoning
3. Be clear and easy to understand
4. Include relevant details from the tool results if applicable

Final Answer:"""
        
        try:
            response = await self.provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=800
            )
            
            return response.strip()
            
        except Exception as e:
            Logger.error(f"Failed to generate final answer: {e}")
            return "I apologize, but I encountered an error while formulating my final answer."