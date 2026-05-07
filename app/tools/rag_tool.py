"""RAG tool for calling rag-api via HTTP."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from ai_service_kit.logging import Logger

from ..models import ToolCall, ToolResult, ToolCallStatus


class RagTool:
    """Tool for performing RAG (Retrieval-Augmented Generation) via HTTP API."""
    
    def __init__(self, api_url: str = "http://localhost:8002"):
        self.api_url = api_url.rstrip("/")
        self.timeout = httpx.Timeout(60.0)  # Longer timeout for generation
    
    @property 
    def name(self) -> str:
        return "rag"
    
    @property
    def description(self) -> str:
        return "Generate answers using retrieval-augmented generation from a knowledge base"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to answer using RAG"
                },
                "collection": {
                    "type": "string", 
                    "description": "Collection name to search in",
                    "default": "default"
                },
                "context_limit": {
                    "type": "integer",
                    "description": "Maximum number of context documents to retrieve",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                },
                "temperature": {
                    "type": "number",
                    "description": "Generation temperature",
                    "default": 0.1,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "include_sources": {
                    "type": "boolean",
                    "description": "Whether to include source citations in the answer",
                    "default": True
                }
            },
            "required": ["question"]
        }
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the RAG tool."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            question = kwargs.get("question")
            if not question:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Missing required parameter: question"
                )
            
            collection = kwargs.get("collection", "default")
            context_limit = kwargs.get("context_limit", 5)
            temperature = kwargs.get("temperature", 0.1)
            include_sources = kwargs.get("include_sources", True)
            
            Logger.debug(f"Executing RAG tool with question: {question}")
            
            # Prepare RAG request
            rag_payload = {
                "question": question,
                "collection": collection,
                "context_limit": context_limit,
                "temperature": temperature,
                "include_sources": include_sources
            }
            
            # Make HTTP request to rag-api
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/query",
                    json=rag_payload
                )
                response.raise_for_status()
                rag_results = response.json()
            
            # Process results
            if "answer" not in rag_results:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolCallStatus.ERROR,
                    error="Invalid response format from RAG API"
                )
            
            answer = rag_results["answer"]
            sources = rag_results.get("sources", [])
            context_used = rag_results.get("context_used", [])
            
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            Logger.debug(f"RAG completed successfully, generated answer with {len(sources)} sources")
            
            # Format the response
            formatted_answer = answer
            if include_sources and sources:
                formatted_answer += "\n\nSources:\n"
                for i, source in enumerate(sources, 1):
                    source_info = source.get("metadata", {}).get("source", f"Source {i}")
                    formatted_answer += f"- {source_info}\n"
            
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.SUCCESS,
                result={
                    "answer": answer,
                    "formatted_answer": formatted_answer,
                    "sources": sources,
                    "context_used": context_used,
                    "question": question
                },
                execution_time_ms=execution_time
            )
            
        except httpx.TimeoutException:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error("RAG tool timed out")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.TIMEOUT,
                error="RAG request timed out",
                execution_time_ms=execution_time
            )
            
        except httpx.HTTPError as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"HTTP error in RAG tool: {e}")
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"HTTP error: {str(e)}",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            Logger.error(f"Unexpected error in RAG tool: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                status=ToolCallStatus.ERROR,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=execution_time
            )
    
    def create_tool_call(self, **kwargs: Any) -> ToolCall:
        """Create a ToolCall object for this execution."""
        return ToolCall(
            tool_name=self.name,
            tool_input=kwargs
        )