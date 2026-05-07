"""Search tool for calling semantic-search-api via HTTP."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from ai_service_kit.logging import Logger

from ..models import ToolCall, ToolResult, ToolCallStatus


class SearchTool:
    """Tool for performing semantic search via HTTP API."""
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0)
    
    @property 
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Perform semantic search to find relevant information from a knowledge base"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant information"
                },
                "collection": {
                    "type": "string", 
                    "description": "Collection name to search in",
                    "default": "default"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score for results",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the search tool."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            query = kwargs.get("query")
            if not query:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Missing required parameter: query"
                )
            
            collection = kwargs.get("collection", "default")
            limit = kwargs.get("limit", 5)
            min_score = kwargs.get("min_score", 0.7)
            
            Logger.debug(f"Executing search tool with query: {query}")
            
            # Prepare search request
            search_payload = {
                "query": query,
                "collection": collection,
                "limit": limit,
                "min_score": min_score
            }
            
            # Make HTTP request to semantic-search-api
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/search",
                    json=search_payload
                )
                response.raise_for_status()
                search_results = response.json()
            
            # Process results
            if "results" not in search_results:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Invalid response format from search API"
                )
            
            results = search_results["results"]
            formatted_results = self._format_results(results)
            
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            Logger.debug(f"Search completed successfully, found {len(results)} results")
            
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.SUCCESS,
                result={
                    "results": results,
                    "formatted_summary": formatted_results,
                    "total_results": len(results),
                    "query": query
                },
                execution_time_ms=execution_time
            )
            
        except httpx.TimeoutException:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error("Search tool timed out")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.TIMEOUT,
                error="Search request timed out",
                execution_time_ms=execution_time
            )
            
        except httpx.HTTPError as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"HTTP error in search tool: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"HTTP error: {str(e)}",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"Unexpected error in search tool: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for display to the agent."""
        if not results:
            return "No relevant results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "No content available")
            score = result.get("score", 0.0)
            source = result.get("metadata", {}).get("source", "Unknown source")
            
            formatted.append(f"{i}. [Score: {score:.3f}] {content}\n   Source: {source}")
        
        return "\n\n".join(formatted)

    def create_tool_call(self, **kwargs: Any) -> ToolCall:
        """Create a ToolCall object for this execution."""
        return ToolCall(
            tool_name=self.name,
            tool_input=kwargs
        )