"""Agent memory management for storing conversation context and reasoning history."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal

from ai_service_kit.logging import Logger

from .models import MemoryEntry


class MemoryManager:
    """Manages both short-term and long-term memory for agents."""
    
    def __init__(self, max_short_term_entries: int = 100, max_long_term_entries: int = 1000):
        self.max_short_term_entries = max_short_term_entries
        self.max_long_term_entries = max_long_term_entries
        
        # In-memory storage (in production, this would use a database)
        self._short_term_memory: Dict[str, MemoryEntry] = {}
        self._long_term_memory: Dict[str, MemoryEntry] = {}
        
        # Usage tracking
        self._access_counts: Dict[str, int] = defaultdict(int)
        
        Logger.info("Memory manager initialized")
    
    async def save_memory(
        self,
        key: str,
        content: str,
        memory_type: Literal["short_term", "long_term"] = "short_term",
        metadata: Dict[str, Any] | None = None
    ) -> None:
        """Save content to memory."""
        try:
            entry = MemoryEntry(
                key=key,
                content=content,
                memory_type=memory_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0
            )
            
            if memory_type == "short_term":
                self._short_term_memory[key] = entry
                await self._cleanup_short_term_memory()
            else:
                self._long_term_memory[key] = entry
                await self._cleanup_long_term_memory()
            
            Logger.debug(f"Saved memory entry: {key} ({memory_type})")
            
        except Exception as e:
            Logger.error(f"Failed to save memory: {e}", exc_info=True)
            raise
    
    async def get_memory(
        self,
        key: str,
        memory_type: Literal["short_term", "long_term", "both"] = "both"
    ) -> str:
        """Retrieve memory content by key."""
        try:
            content_parts = []
            
            if memory_type in ["short_term", "both"]:
                if key in self._short_term_memory:
                    entry = self._short_term_memory[key]
                    entry.access_count += 1
                    entry.updated_at = datetime.now()
                    content_parts.append(f"[Short-term]: {entry.content}")
                    Logger.debug(f"Retrieved short-term memory: {key}")
            
            if memory_type in ["long_term", "both"]:
                if key in self._long_term_memory:
                    entry = self._long_term_memory[key]
                    entry.access_count += 1
                    entry.updated_at = datetime.now()
                    content_parts.append(f"[Long-term]: {entry.content}")
                    Logger.debug(f"Retrieved long-term memory: {key}")
            
            return "\n".join(content_parts) if content_parts else ""
            
        except Exception as e:
            Logger.error(f"Failed to retrieve memory: {e}", exc_info=True)
            return ""
    
    async def update_memory(
        self,
        key: str,
        new_content: str,
        memory_type: Literal["short_term", "long_term"] = "short_term"
    ) -> bool:
        """Update existing memory entry."""
        try:
            memory_store = (
                self._short_term_memory if memory_type == "short_term" 
                else self._long_term_memory
            )
            
            if key in memory_store:
                entry = memory_store[key]
                entry.content = new_content
                entry.updated_at = datetime.now()
                Logger.debug(f"Updated memory entry: {key} ({memory_type})")
                return True
            else:
                Logger.warning(f"Memory entry not found for update: {key} ({memory_type})")
                return False
                
        except Exception as e:
            Logger.error(f"Failed to update memory: {e}", exc_info=True)
            return False
    
    async def delete_memory(
        self,
        key: str,
        memory_type: Literal["short_term", "long_term", "both"] = "both"
    ) -> bool:
        """Delete memory entry by key."""
        try:
            deleted = False
            
            if memory_type in ["short_term", "both"]:
                if key in self._short_term_memory:
                    del self._short_term_memory[key]
                    deleted = True
                    Logger.debug(f"Deleted short-term memory: {key}")
            
            if memory_type in ["long_term", "both"]:
                if key in self._long_term_memory:
                    del self._long_term_memory[key]
                    deleted = True
                    Logger.debug(f"Deleted long-term memory: {key}")
            
            return deleted
            
        except Exception as e:
            Logger.error(f"Failed to delete memory: {e}", exc_info=True)
            return False
    
    async def list_memory_keys(
        self,
        memory_type: Literal["short_term", "long_term", "both"] = "both"
    ) -> List[str]:
        """List all memory keys."""
        try:
            keys = []
            
            if memory_type in ["short_term", "both"]:
                keys.extend(self._short_term_memory.keys())
            
            if memory_type in ["long_term", "both"]:
                keys.extend(self._long_term_memory.keys())
            
            return list(set(keys))  # Remove duplicates
            
        except Exception as e:
            Logger.error(f"Failed to list memory keys: {e}", exc_info=True)
            return []
    
    async def search_memory(
        self,
        search_term: str,
        memory_type: Literal["short_term", "long_term", "both"] = "both"
    ) -> List[MemoryEntry]:
        """Search for memory entries containing a term."""
        try:
            results = []
            
            memories_to_search = []
            if memory_type in ["short_term", "both"]:
                memories_to_search.extend(self._short_term_memory.values())
            if memory_type in ["long_term", "both"]:
                memories_to_search.extend(self._long_term_memory.values())
            
            search_term_lower = search_term.lower()
            
            for entry in memories_to_search:
                if (search_term_lower in entry.content.lower() or 
                    search_term_lower in entry.key.lower()):
                    results.append(entry)
            
            # Sort by access count and recency
            results.sort(key=lambda x: (x.access_count, x.updated_at), reverse=True)
            
            Logger.debug(f"Memory search for '{search_term}' found {len(results)} results")
            return results
            
        except Exception as e:
            Logger.error(f"Failed to search memory: {e}", exc_info=True)
            return []
    
    async def promote_to_long_term(self, key: str) -> bool:
        """Promote a short-term memory entry to long-term memory."""
        try:
            if key in self._short_term_memory:
                entry = self._short_term_memory[key]
                entry.memory_type = "long_term"
                entry.updated_at = datetime.now()
                
                self._long_term_memory[key] = entry
                del self._short_term_memory[key]
                
                await self._cleanup_long_term_memory()
                
                Logger.debug(f"Promoted memory to long-term: {key}")
                return True
            else:
                Logger.warning(f"Short-term memory entry not found for promotion: {key}")
                return False
                
        except Exception as e:
            Logger.error(f"Failed to promote memory: {e}", exc_info=True)
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            return {
                "short_term_count": len(self._short_term_memory),
                "long_term_count": len(self._long_term_memory),
                "max_short_term": self.max_short_term_entries,
                "max_long_term": self.max_long_term_entries,
                "total_entries": len(self._short_term_memory) + len(self._long_term_memory),
                "most_accessed_short_term": self._get_most_accessed("short_term"),
                "most_accessed_long_term": self._get_most_accessed("long_term"),
                "oldest_entry": self._get_oldest_entry(),
                "newest_entry": self._get_newest_entry(),
            }
        except Exception as e:
            Logger.error(f"Failed to get memory stats: {e}", exc_info=True)
            return {}
    
    async def clear_memory(
        self,
        memory_type: Literal["short_term", "long_term", "both"] = "both"
    ) -> None:
        """Clear memory entries."""
        try:
            if memory_type in ["short_term", "both"]:
                self._short_term_memory.clear()
                Logger.info("Cleared short-term memory")
            
            if memory_type in ["long_term", "both"]:
                self._long_term_memory.clear()
                Logger.info("Cleared long-term memory")
                
        except Exception as e:
            Logger.error(f"Failed to clear memory: {e}", exc_info=True)
    
    async def _cleanup_short_term_memory(self) -> None:
        """Remove oldest short-term memory entries if limit exceeded."""
        if len(self._short_term_memory) > self.max_short_term_entries:
            # Sort by access count and age, remove least used and oldest
            entries = list(self._short_term_memory.items())
            entries.sort(key=lambda x: (x[1].access_count, x[1].updated_at))
            
            entries_to_remove = len(entries) - self.max_short_term_entries
            for i in range(entries_to_remove):
                key, entry = entries[i]
                del self._short_term_memory[key]
                Logger.debug(f"Removed old short-term memory entry: {key}")
    
    async def _cleanup_long_term_memory(self) -> None:
        """Remove oldest long-term memory entries if limit exceeded."""
        if len(self._long_term_memory) > self.max_long_term_entries:
            # Sort by access count and age, remove least used and oldest
            entries = list(self._long_term_memory.items())
            entries.sort(key=lambda x: (x[1].access_count, x[1].updated_at))
            
            entries_to_remove = len(entries) - self.max_long_term_entries
            for i in range(entries_to_remove):
                key, entry = entries[i]
                del self._long_term_memory[key]
                Logger.debug(f"Removed old long-term memory entry: {key}")
    
    def _get_most_accessed(self, memory_type: str) -> Dict[str, Any]:
        """Get the most accessed memory entry for a given type."""
        memory_store = (
            self._short_term_memory if memory_type == "short_term" 
            else self._long_term_memory
        )
        
        if not memory_store:
            return {"key": None, "access_count": 0}
        
        most_accessed = max(memory_store.values(), key=lambda x: x.access_count)
        return {
            "key": most_accessed.key,
            "access_count": most_accessed.access_count,
            "created_at": most_accessed.created_at.isoformat(),
        }
    
    def _get_oldest_entry(self) -> Dict[str, Any]:
        """Get the oldest memory entry across all types."""
        all_entries = list(self._short_term_memory.values()) + list(self._long_term_memory.values())
        
        if not all_entries:
            return {"key": None, "created_at": None}
        
        oldest = min(all_entries, key=lambda x: x.created_at)
        return {
            "key": oldest.key,
            "memory_type": oldest.memory_type,
            "created_at": oldest.created_at.isoformat(),
        }
    
    def _get_newest_entry(self) -> Dict[str, Any]:
        """Get the newest memory entry across all types."""
        all_entries = list(self._short_term_memory.values()) + list(self._long_term_memory.values())
        
        if not all_entries:
            return {"key": None, "created_at": None}
        
        newest = max(all_entries, key=lambda x: x.created_at)
        return {
            "key": newest.key,
            "memory_type": newest.memory_type,
            "created_at": newest.created_at.isoformat(),
        }