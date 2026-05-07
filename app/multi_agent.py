"""Multi-agent workflow implementing planner → worker → reviewer pattern."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, List

from ai_service_kit.logging import Logger

from .agent import ReactAgent
from .config import get_settings
from .guardrails import GuardrailsManager
from .memory import MemoryManager
from .models import (
    MultiAgentRequest, MultiAgentResult, MultiAgentStep, MultiAgentPlan,
    ReasoningStep, ToolCall, ToolResult, AgentRequest
)
from .router import ModelRouter
from .semantic_cache import SemanticCache


class MultiAgentOrchestrator:
    """Orchestrates a multi-agent workflow with planner, worker, and reviewer agents."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize component agents
        self.planner_agent = ReactAgent()
        self.worker_agent = ReactAgent() 
        self.reviewer_agent = ReactAgent()
        
        # Initialize supporting systems
        self.router = ModelRouter()
        self.cache = SemanticCache()
        self.guardrails = GuardrailsManager()
        self.memory = MemoryManager()
        
        Logger.info("Multi-agent orchestrator initialized")
    
    async def execute(self, request: MultiAgentRequest) -> MultiAgentResult:
        """Execute the full multi-agent workflow."""
        start_time = time.time()
        
        Logger.info(f"Starting multi-agent execution for query: {request.query[:100]}...")
        
        try:
            # Check semantic cache first
            cached_response = await self.cache.get_cached_response(request.query)
            if cached_response:
                Logger.info("Multi-agent query served from semantic cache")
                # Create a simplified result from cache
                return self._create_cached_result(request, cached_response, start_time)
            
            # Validate request through guardrails
            validation_result = await self.guardrails.validate_input(request.query)
            if not validation_result.passed:
                return self._create_error_result(
                    request, 
                    f"Input validation failed: {validation_result.message}",
                    start_time
                )
            
            steps = []
            all_tool_calls = []
            all_tool_results = []
            
            # Phase 1: Planning
            Logger.debug("Starting planning phase")
            plan, planning_step = await self._execute_planning_phase(request)
            steps.append(planning_step)
            
            if not plan:
                return self._create_error_result(
                    request,
                    "Failed to create execution plan",
                    start_time
                )
            
            # Phase 2: Execution with iterations
            final_answer = None
            for iteration in range(1, request.max_iterations + 1):
                Logger.debug(f"Starting iteration {iteration}/{request.max_iterations}")
                
                # Worker phase
                worker_step, worker_calls, worker_results = await self._execute_worker_phase(
                    request, plan, iteration
                )
                steps.append(worker_step)
                all_tool_calls.extend(worker_calls)
                all_tool_results.extend(worker_results)
                
                # Reviewer phase
                reviewer_step, final_answer, needs_refinement = await self._execute_reviewer_phase(
                    request, plan, steps, iteration
                )
                steps.append(reviewer_step)
                
                # Check if we're done
                if not needs_refinement or iteration == request.max_iterations:
                    Logger.debug(f"Multi-agent workflow completed after {iteration} iterations")
                    break
                
                # Check timeout
                if time.time() - start_time > request.timeout_seconds:
                    Logger.warning("Multi-agent execution timed out")
                    break
            
            # Validate final output
            if final_answer:
                output_validation = await self.guardrails.validate_output(final_answer)
                if not output_validation.passed:
                    final_answer = output_validation.filtered_content or "I cannot provide a response due to content policy restrictions."
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Cache the result
            if final_answer:
                await self.cache.cache_response(
                    request.query,
                    final_answer,
                    metadata={"multi_agent": True, "iterations": iteration}
                )
            
            result = MultiAgentResult(
                query=request.query,
                plan=plan,
                steps=steps,
                final_answer=final_answer or "Unable to generate final answer",
                iterations=iteration,
                execution_time_ms=execution_time_ms,
                success=final_answer is not None
            )
            
            Logger.info(f"Multi-agent execution completed successfully in {execution_time_ms}ms with {iteration} iterations")
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            Logger.error(f"Multi-agent execution failed: {e}", exc_info=True)
            
            return MultiAgentResult(
                query=request.query,
                plan=plan if 'plan' in locals() else MultiAgentPlan(
                    task=request.query,
                    subtasks=[],
                    reasoning="Planning failed",
                    estimated_complexity="medium"
                ),
                steps=steps if 'steps' in locals() else [],
                final_answer="I encountered an error while processing your request.",
                iterations=0,
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
    
    async def _execute_planning_phase(self, request: MultiAgentRequest) -> tuple[MultiAgentPlan | None, MultiAgentStep]:
        """Execute the planning phase with the planner agent."""
        
        planning_prompt = f"""You are an expert task planner. Break down this complex request into specific, actionable subtasks.

User Request: {request.query}

Available Tools: {', '.join(request.tools)}

Your task:
1. Analyze the request complexity and scope
2. Break it down into 3-7 specific subtasks that can be executed independently
3. Order the subtasks logically
4. Estimate overall complexity

Provide your response in this exact format:

ANALYSIS:
[Your analysis of the request and approach]

SUBTASKS:
1. [First specific subtask]
2. [Second specific subtask]
...

COMPLEXITY: [low/medium/high]

REASONING:
[Explain your planning decisions]"""

        try:
            # Use expensive model for planning
            provider, route_decision = await self.router.route_request(
                planning_prompt,
                complexity_hint="complex planning task",
                force_expensive=True
            )
            
            if provider is None:
                raise RuntimeError("Planning provider not available")
            
            planning_response = await provider.generate(
                prompt=planning_prompt,
                temperature=request.temperature,
                max_tokens=1000
            )
            planning_response_text = (
                planning_response.content if hasattr(planning_response, "content") else str(planning_response)
            )
            
            # Parse the planning response
            plan = self._parse_planning_response(request.query, planning_response_text)
            
            step = MultiAgentStep(
                agent_role="planner",
                step=1,
                input=request.query,
                output=planning_response_text,
                reasoning=[],
                tool_calls=[],
                tool_results=[]
            )
            
            return plan, step
            
        except Exception as e:
            Logger.error(f"Planning phase failed: {e}")
            
            error_step = MultiAgentStep(
                agent_role="planner",
                step=1,
                input=request.query,
                output=f"Planning failed: {str(e)}",
                reasoning=[],
                tool_calls=[],
                tool_results=[]
            )
            
            return None, error_step
    
    async def _execute_worker_phase(
        self,
        request: MultiAgentRequest,
        plan: MultiAgentPlan,
        iteration: int
    ) -> tuple[MultiAgentStep, List[ToolCall], List[ToolResult]]:
        """Execute the worker phase to complete the planned subtasks."""
        
        # Create worker prompt
        subtasks_text = "\n".join([f"{i+1}. {task}" for i, task in enumerate(plan.subtasks)])
        
        worker_prompt = f"""You are a specialized worker agent executing planned subtasks.

Original Query: {request.query}

Plan Overview:
{plan.reasoning}

Subtasks to Execute:
{subtasks_text}

Your job is to work through these subtasks systematically, using available tools when needed.
Focus on gathering information, performing analysis, and producing concrete results for each subtask.

Available tools: {', '.join(request.tools)}

Execute the subtasks step by step and provide your findings."""

        try:
            # Create agent request for worker
            worker_request = AgentRequest(
                query=worker_prompt,
                tools=request.tools,
                max_steps=15,  # Allow more steps for complex work
                temperature=request.temperature,
                timeout_seconds=min(300, request.timeout_seconds // 2)  # Allocate time
            )
            
            # Execute worker agent
            worker_response = await self.worker_agent.execute(worker_request)
            
            step = MultiAgentStep(
                agent_role="worker",
                step=len([s for s in [] if s.agent_role == "worker"]) + 1,
                input=worker_prompt,
                output=worker_response.answer,
                reasoning=worker_response.trace,
                tool_calls=worker_response.tool_calls,
                tool_results=worker_response.tool_results
            )
            
            return step, worker_response.tool_calls, worker_response.tool_results
            
        except Exception as e:
            Logger.error(f"Worker phase failed: {e}")
            
            error_step = MultiAgentStep(
                agent_role="worker",
                step=iteration * 2,
                input=worker_prompt,
                output=f"Worker execution failed: {str(e)}",
                reasoning=[],
                tool_calls=[],
                tool_results=[]
            )
            
            return error_step, [], []
    
    async def _execute_reviewer_phase(
        self,
        request: MultiAgentRequest,
        plan: MultiAgentPlan,
        all_steps: List[MultiAgentStep],
        iteration: int
    ) -> tuple[MultiAgentStep, str | None, bool]:
        """Execute the reviewer phase to validate and potentially refine the work."""
        
        # Gather worker outputs
        worker_outputs = [step.output for step in all_steps if step.agent_role == "worker"]
        worker_summary = "\n\n".join(worker_outputs)
        
        reviewer_prompt = f"""You are a quality reviewer agent. Review the work completed and provide a final answer.

Original Query: {request.query}

Execution Plan:
{plan.reasoning}
Planned subtasks: {', '.join(plan.subtasks)}

Worker Results:
{worker_summary}

Your tasks:
1. Review if the worker results adequately address the original query
2. Identify any gaps or areas needing improvement
3. Provide a comprehensive final answer OR request specific refinements

If the work is complete and satisfactory, provide a final answer starting with "FINAL ANSWER:"
If refinements are needed, start with "REFINEMENT NEEDED:" and specify what needs improvement.

Be thorough and ensure the response fully addresses the user's original request."""

        try:
            # Use expensive model for review
            provider, route_decision = await self.router.route_request(
                reviewer_prompt,
                complexity_hint="complex review and synthesis",
                force_expensive=True
            )
            
            if provider is None:
                raise RuntimeError("Review provider not available")
            
            reviewer_response = await provider.generate(
                prompt=reviewer_prompt,
                temperature=request.temperature,
                max_tokens=1200
            )
            reviewer_response_text = (
                reviewer_response.content if hasattr(reviewer_response, "content") else str(reviewer_response)
            )
            
            # Parse reviewer decision
            needs_refinement = reviewer_response_text.strip().startswith("REFINEMENT NEEDED:")
            
            if needs_refinement:
                final_answer = None
            else:
                # Extract final answer
                if "FINAL ANSWER:" in reviewer_response_text:
                    final_answer = reviewer_response_text.split("FINAL ANSWER:", 1)[1].strip()
                else:
                    final_answer = reviewer_response_text.strip()
            
            step = MultiAgentStep(
                agent_role="reviewer",
                step=len(all_steps) + 1,
                input=reviewer_prompt,
                output=reviewer_response_text,
                reasoning=[],
                tool_calls=[],
                tool_results=[]
            )
            
            return step, final_answer, needs_refinement
            
        except Exception as e:
            Logger.error(f"Reviewer phase failed: {e}")
            
            error_step = MultiAgentStep(
                agent_role="reviewer",
                step=len(all_steps) + 1,
                input=reviewer_prompt,
                output=f"Review failed: {str(e)}",
                reasoning=[],
                tool_calls=[],
                tool_results=[]
            )
            
            return error_step, None, False
    
    def _parse_planning_response(self, original_query: str, response: str) -> MultiAgentPlan | None:
        """Parse the planner's response into a structured plan."""
        try:
            lines = response.strip().split('\n')
            
            analysis = ""
            subtasks = []
            complexity = "medium"
            reasoning = ""
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("ANALYSIS:"):
                    current_section = "analysis"
                    continue
                elif line.startswith("SUBTASKS:"):
                    current_section = "subtasks"
                    continue
                elif line.startswith("COMPLEXITY:"):
                    complexity = line.split(":", 1)[1].strip().lower()
                    if complexity not in ["low", "medium", "high"]:
                        complexity = "medium"
                    current_section = None
                    continue
                elif line.startswith("REASONING:"):
                    current_section = "reasoning"
                    continue
                
                if current_section == "analysis":
                    analysis += line + " "
                elif current_section == "subtasks":
                    if line and (line.startswith(tuple(f"{i}." for i in range(1, 10)))):
                        # Extract subtask text
                        task = line.split(".", 1)[1].strip()
                        subtasks.append(task)
                elif current_section == "reasoning":
                    reasoning += line + " "
            
            if not subtasks:
                # Fallback parsing
                subtasks = [f"Address: {original_query}"]
            
            return MultiAgentPlan(
                task=original_query,
                subtasks=subtasks,
                reasoning=reasoning.strip() or analysis.strip() or "Analyze and complete the requested task",
                estimated_complexity=complexity
            )
            
        except Exception as e:
            Logger.error(f"Failed to parse planning response: {e}")
            return None
    
    def _create_cached_result(self, request: MultiAgentRequest, cached_response: str, start_time: float) -> MultiAgentResult:
        """Create a result object for cached responses."""
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return MultiAgentResult(
            query=request.query,
            plan=MultiAgentPlan(
                task=request.query,
                subtasks=["Retrieved from cache"],
                reasoning="Response served from semantic cache",
                estimated_complexity="low"
            ),
            steps=[
                MultiAgentStep(
                    agent_role="planner",
                    step=1,
                    input=request.query,
                    output="Serving cached response",
                    reasoning=[],
                    tool_calls=[],
                    tool_results=[]
                )
            ],
            final_answer=cached_response,
            iterations=0,
            execution_time_ms=execution_time_ms,
            success=True
        )
    
    def _create_error_result(self, request: MultiAgentRequest, error_message: str, start_time: float) -> MultiAgentResult:
        """Create an error result object."""
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return MultiAgentResult(
            query=request.query,
            plan=MultiAgentPlan(
                task=request.query,
                subtasks=[],
                reasoning="Failed to execute",
                estimated_complexity="medium"
            ),
            steps=[],
            final_answer="I cannot process this request.",
            iterations=0,
            execution_time_ms=execution_time_ms,
            success=False,
            error=error_message
        )