from __future__ import annotations

from fastapi.encoders import jsonable_encoder
from ai_service_kit.health import SimpleHealthResponse, check_health, get_diagnostics, get_metrics, ping_service
from ai_service_kit.logging import setup_enhanced_logging, LoggingMiddleware, Logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import ReactAgent
from .bootstrap import build_service_context, debug_snapshot
from .config import get_settings
from .models import (
    AgentRequest, AgentResponse, MultiAgentRequest, MultiAgentResult,
    ToolsResponse, TraceResponse, ErrorResponse, ToolInfo
)
from .multi_agent import MultiAgentOrchestrator
from .router import ModelRouter
from .semantic_cache import SemanticCache
from .tools import AVAILABLE_TOOLS, get_tool_by_name


def create_app() -> FastAPI:
    # Setup enhanced logging first
    setup_enhanced_logging()
    
    settings = get_settings()
    service_context = build_service_context(settings)

    app = FastAPI(title=settings.app_name, debug=settings.app_debug)
    app.state.settings = settings
    app.state.service_context = service_context
    
    # Initialize agent components
    app.state.react_agent = ReactAgent()
    app.state.multi_agent = MultiAgentOrchestrator()
    app.state.router = ModelRouter()
    app.state.cache = SemanticCache()
    app.state.last_execution = None  # Store last execution for debug trace
    
    Logger.info(f"Starting {settings.app_name} v{settings.app_version} in {settings.app_env} mode")

    # Add logging middleware first (after CORS)
    if settings.enable_cors and settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add logging middleware for request correlation and tracking
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, object]:
        try:
            Logger.debug("Ping endpoint called")
            result = ping_service(app.state.service_context)
            Logger.debug("Ping endpoint completed successfully")
            return jsonable_encoder(result)
        except Exception as e:
            Logger.error(f"Ping endpoint failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health")
    async def health() -> dict[str, object]:
        try:
            Logger.debug("Health check endpoint called")
            result = await check_health(app.state.service_context)
            # Convert to dict for logging and jsonable_encoder
            result_dict = jsonable_encoder(result)
            status = result_dict.get('status', 'unknown')
            Logger.info(f"Health check completed with status: {status}")
            return result_dict
        except Exception as e:
            Logger.error(f"Health check failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Health check failed")

    @app.get("/diagnostics")
    async def diagnostics() -> dict[str, object]:
        try:
            Logger.debug("Diagnostics endpoint called")
            result = await get_diagnostics(app.state.service_context)
            # Convert to dict for logging and jsonable_encoder
            result_dict = jsonable_encoder(result)
            total_checks = result_dict.get('summary', {}).get('total_checks', 0)
            Logger.info(f"Diagnostics completed with {total_checks} checks")
            return result_dict
        except Exception as e:
            Logger.error(f"Diagnostics failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Diagnostics failed")

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        try:
            Logger.debug("Metrics endpoint called")
            result = get_metrics(app.state.service_context)
            Logger.debug("Metrics endpoint completed successfully")
            return jsonable_encoder(result)
        except Exception as e:
            Logger.error(f"Metrics collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Metrics collection failed")

    @app.get("/debug/config")
    async def debug_config() -> dict[str, object]:
        try:
            Logger.debug("Debug config endpoint called")
            result = {
                "app": dict(settings.masked_debug_config()),
                "bootstrap": debug_snapshot(app.state.service_context),
            }
            Logger.debug("Debug config endpoint completed successfully")
            return jsonable_encoder(result)
        except Exception as e:
            Logger.error(f"Debug config failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Debug config failed")

    # Agent Endpoints
    
    @app.post("/agent/query", response_model=AgentResponse)
    async def agent_query(request: AgentRequest) -> AgentResponse:
        """Execute a single-agent ReAct reasoning loop."""
        try:
            Logger.info(f"Agent query request: {request.query[:100]}...")
            
            result = await app.state.react_agent.execute(request)
            app.state.last_execution = result  # Store for debug trace
            
            Logger.info(f"Agent query completed successfully: {result.success}")
            return result
            
        except Exception as e:
            Logger.error(f"Agent query failed: {e}", exc_info=True)
            error_response = AgentResponse(
                query=request.query,
                answer="I encountered an error while processing your request.",
                trace=[],
                tool_calls=[],
                tool_results=[],
                total_steps=0,
                execution_time_ms=0,
                success=False,
                error=str(e)
            )
            app.state.last_execution = error_response
            return error_response

    @app.post("/agent/multi", response_model=MultiAgentResult)
    async def agent_multi(request: MultiAgentRequest) -> MultiAgentResult:
        """Execute a multi-agent planner → worker → reviewer workflow."""
        try:
            Logger.info(f"Multi-agent request: {request.query[:100]}...")
            
            result = await app.state.multi_agent.execute(request)
            app.state.last_execution = result  # Store for debug trace
            
            Logger.info(f"Multi-agent query completed successfully: {result.success}")
            return result
            
        except Exception as e:
            Logger.error(f"Multi-agent query failed: {e}", exc_info=True)
            from .models import MultiAgentPlan
            error_response = MultiAgentResult(
                query=request.query,
                plan=MultiAgentPlan(
                    task=request.query,
                    subtasks=[],
                    reasoning="Execution failed",
                    estimated_complexity="medium"
                ),
                steps=[],
                final_answer="I encountered an error while processing your request.",
                iterations=0,
                execution_time_ms=0,
                success=False,
                error=str(e)
            )
            app.state.last_execution = error_response
            return error_response

    @app.get("/agent/tools", response_model=ToolsResponse)
    async def agent_tools() -> ToolsResponse:
        """List available tools for agent execution."""
        try:
            Logger.debug("Agent tools request")
            
            tools = []
            for tool_name in AVAILABLE_TOOLS.keys():
                tool = get_tool_by_name(tool_name)
                tool_info = ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema
                )
                tools.append(tool_info)
            
            result = ToolsResponse(
                tools=tools,
                total_count=len(tools)
            )
            
            Logger.debug(f"Returning {len(tools)} available tools")
            return result
            
        except Exception as e:
            Logger.error(f"Agent tools request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve tools")

    @app.get("/agent/trace", response_model=TraceResponse)
    async def agent_trace() -> TraceResponse:
        """Debug endpoint to inspect last agent execution."""
        try:
            Logger.debug("Agent trace request")
            
            # Get cache stats
            cache_stats = await app.state.cache.get_cache_stats()
            
            # Get memory stats (if last execution stored memory)
            memory_stats = {
                "last_execution_stored": app.state.last_execution is not None,
                "execution_type": type(app.state.last_execution).__name__ if app.state.last_execution else None
            }
            
            # Get tool usage stats
            tool_stats = {}
            if app.state.last_execution:
                if hasattr(app.state.last_execution, 'tool_calls'):
                    # Single agent result
                    for tool_call in app.state.last_execution.tool_calls:
                        tool_name = tool_call.tool_name
                        tool_stats[tool_name] = tool_stats.get(tool_name, 0) + 1
                elif hasattr(app.state.last_execution, 'steps'):
                    # Multi-agent result
                    for step in app.state.last_execution.steps:
                        for tool_call in step.tool_calls:
                            tool_name = tool_call.tool_name
                            tool_stats[tool_name] = tool_stats.get(tool_name, 0) + 1
            
            result = TraceResponse(
                last_execution=app.state.last_execution,
                cache_stats=cache_stats,
                memory_stats=memory_stats,
                tool_stats=tool_stats
            )
            
            Logger.debug("Agent trace completed successfully")
            return result
            
        except Exception as e:
            Logger.error(f"Agent trace failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve trace")
    
    # Additional utility endpoints
    
    @app.post("/debug/cache/clear")
    async def clear_cache():
        """Clear the semantic cache (debug endpoint)."""
        try:
            Logger.debug("Clear cache request")
            await app.state.cache.clear_cache()
            Logger.info("Semantic cache cleared via debug endpoint")
            return {"status": "success", "message": "Cache cleared"}
        except Exception as e:
            Logger.error(f"Clear cache failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    @app.get("/debug/router/stats")
    async def router_stats():
        """Get model router statistics (debug endpoint)."""
        try:
            Logger.debug("Router stats request")
            stats = await app.state.router.get_router_stats()
            Logger.debug("Router stats completed successfully")
            return stats
        except Exception as e:
            Logger.error(f"Router stats failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to get router stats")
    
    @app.get("/debug/router/health")
    async def router_health():
        """Check health of model routing providers (debug endpoint)."""
        try:
            Logger.debug("Router health request")
            health_results = await app.state.router.health_check()
            Logger.debug("Router health check completed successfully")
            return health_results
        except Exception as e:
            Logger.error(f"Router health check failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to check router health")

    return app


app = create_app()