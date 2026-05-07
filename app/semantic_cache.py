"""Semantic caching using vector similarity for agent responses."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
from ai_service_kit.logging import Logger
from ai_service_kit.providers import ProviderFactory

from .config import get_settings
from .models import CacheEntry


class SemanticCache:
    """Vector similarity-based caching for agent responses."""
    
    def __init__(self):
        self.settings = get_settings()
        self._embedding_provider = None
        self._cache: Dict[str, CacheEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Cache configuration
        self.similarity_threshold = self.settings.semantic_cache_threshold
        self.max_entries = self.settings.semantic_cache_max_entries
        self.enabled = self.settings.semantic_cache_enabled
        
        # Performance tracking
        self._hits = 0
        self._misses = 0
        self._total_queries = 0
        
        Logger.info(f"Semantic cache initialized (enabled: {self.enabled}, threshold: {self.similarity_threshold})")
    
    async def _ensure_embedding_provider(self):
        """Ensure the embedding provider is initialized."""
        if self._embedding_provider is None and self.enabled:
            try:
                factory = ProviderFactory()
                
                provider_config = {
                    "api_key": self.settings.selected_provider_api_key(),
                }
                
                # Add any additional config
                settings_config = self.settings.provider_config()
                provider_config.update(settings_config)
                
                self._embedding_provider = factory.create_provider(
                    provider_name=self.settings.provider_type,
                    config=provider_config
                )
                Logger.debug(f"Initialized embedding provider: {self.settings.provider_type}")
            except Exception as e:
                Logger.error(f"Failed to initialize embedding provider: {e}")
                self.enabled = False
    
    async def get_cached_response(self, query: str) -> str | None:
        """Get cached response for a query if similarity is above threshold."""
        if not self.enabled:
            return None
        
        self._total_queries += 1
        
        try:
            await self._ensure_embedding_provider()
            
            if not self._embedding_provider:
                return None
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return None
            
            # Find most similar cached entry
            best_match, best_similarity = await self._find_best_match(query_embedding)
            
            if best_match and best_similarity >= self.similarity_threshold:
                # Update access statistics
                best_match.access_count += 1
                
                self._hits += 1
                Logger.debug(f"Cache hit for query (similarity: {best_similarity:.3f}): {query[:50]}...")
                
                return best_match.response
            else:
                self._misses += 1
                Logger.debug(f"Cache miss for query (best similarity: {best_similarity:.3f}): {query[:50]}...")
                return None
                
        except Exception as e:
            Logger.error(f"Error retrieving cached response: {e}", exc_info=True)
            return None
    
    async def cache_response(
        self,
        query: str,
        response: str,
        metadata: Dict[str, Any] | None = None
    ) -> bool:
        """Cache a query-response pair."""
        if not self.enabled:
            return False
        
        try:
            await self._ensure_embedding_provider()
            
            if not self._embedding_provider:
                return False
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return False
            
            # Create cache entry
            cache_key = self._generate_cache_key(query)
            cache_entry = CacheEntry(
                query=query,
                embedding=query_embedding.tolist(),
                response=response,
                metadata=metadata or {},
                created_at=datetime.now(),
                access_count=0
            )
            
            # Store in cache
            self._cache[cache_key] = cache_entry
            self._embeddings[cache_key] = query_embedding
            
            # Cleanup if necessary
            await self._cleanup_cache()
            
            Logger.debug(f"Cached response for query: {query[:50]}...")
            return True
            
        except Exception as e:
            Logger.error(f"Error caching response: {e}", exc_info=True)
            return False
    
    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding for a text string."""
        try:
            if not self._embedding_provider:
                return None
            
            # Use the embedding method from ai-service-kit
            embedding_result = await self._embedding_provider.embed(text)
            
            if hasattr(embedding_result, 'embedding'):
                return np.array(embedding_result.embedding)
            elif isinstance(embedding_result, (list, np.ndarray)):
                return np.array(embedding_result)
            else:
                Logger.error(f"Unexpected embedding result type: {type(embedding_result)}")
                return None
                
        except Exception as e:
            Logger.error(f"Failed to get embedding: {e}")
            return None
    
    async def _find_best_match(self, query_embedding: np.ndarray) -> Tuple[CacheEntry | None, float]:
        """Find the cached entry with highest similarity to the query."""
        if not self._embeddings:
            return None, 0.0
        
        best_entry = None
        best_similarity = 0.0
        
        try:
            for cache_key, cached_embedding in self._embeddings.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = self._cache.get(cache_key)
            
            return best_entry, best_similarity
            
        except Exception as e:
            Logger.error(f"Error finding best match: {e}")
            return None, 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            Logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query."""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    async def _cleanup_cache(self):
        """Remove old entries if cache exceeds maximum size."""
        if len(self._cache) <= self.max_entries:
            return
        
        try:
            # Sort by access count and age (least recently used)
            entries = list(self._cache.items())
            entries.sort(key=lambda x: (x[1].access_count, x[1].created_at))
            
            # Remove oldest, least accessed entries
            entries_to_remove = len(entries) - self.max_entries
            
            for i in range(entries_to_remove):
                cache_key, _ = entries[i]
                del self._cache[cache_key]
                del self._embeddings[cache_key]
            
            Logger.debug(f"Cleaned up {entries_to_remove} cache entries")
            
        except Exception as e:
            Logger.error(f"Error during cache cleanup: {e}")
    
    async def clear_cache(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._embeddings.clear()
        Logger.info("Semantic cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = (self._hits / self._total_queries * 100) if self._total_queries > 0 else 0
        
        # Calculate cache age statistics
        if self._cache:
            ages = [(datetime.now() - entry.created_at).total_seconds() for entry in self._cache.values()]
            avg_age = sum(ages) / len(ages)
            max_age = max(ages)
            min_age = min(ages)
        else:
            avg_age = max_age = min_age = 0
        
        # Calculate access statistics
        if self._cache:
            access_counts = [entry.access_count for entry in self._cache.values()]
            avg_access = sum(access_counts) / len(access_counts)
            max_access = max(access_counts)
        else:
            avg_access = max_access = 0
        
        return {
            "enabled": self.enabled,
            "total_entries": len(self._cache),
            "max_entries": self.max_entries,
            "similarity_threshold": self.similarity_threshold,
            "total_queries": self._total_queries,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "avg_entry_age_seconds": round(avg_age, 2),
            "max_entry_age_seconds": round(max_age, 2),
            "min_entry_age_seconds": round(min_age, 2),
            "avg_access_count": round(avg_access, 2),
            "max_access_count": max_access,
            "memory_usage_estimate_mb": self._estimate_memory_usage(),
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the cache in MB."""
        try:
            total_size = 0
            
            # Estimate size of cache entries
            for entry in self._cache.values():
                total_size += len(entry.query.encode('utf-8'))
                total_size += len(entry.response.encode('utf-8'))
                total_size += len(str(entry.metadata).encode('utf-8'))
                total_size += len(entry.embedding) * 4  # Assuming 4 bytes per float
            
            # Estimate size of embeddings
            for embedding in self._embeddings.values():
                total_size += embedding.nbytes
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            Logger.error(f"Error estimating memory usage: {e}")
            return 0.0
    
    async def invalidate_similar(self, query: str, threshold: float | None = None) -> int:
        """Invalidate cache entries similar to the given query."""
        if not self.enabled or not self._embedding_provider:
            return 0
        
        threshold = threshold or self.similarity_threshold
        
        try:
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return 0
            
            keys_to_remove = []
            
            for cache_key, cached_embedding in self._embeddings.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity >= threshold:
                    keys_to_remove.append(cache_key)
            
            # Remove the entries
            for key in keys_to_remove:
                del self._cache[key]
                del self._embeddings[key]
            
            Logger.info(f"Invalidated {len(keys_to_remove)} similar cache entries")
            return len(keys_to_remove)
            
        except Exception as e:
            Logger.error(f"Error invalidating similar entries: {e}")
            return 0
    
    async def preload_cache(self, query_response_pairs: List[Tuple[str, str]]) -> int:
        """Preload the cache with multiple query-response pairs."""
        if not self.enabled:
            return 0
        
        loaded_count = 0
        
        for query, response in query_response_pairs:
            try:
                success = await self.cache_response(query, response)
                if success:
                    loaded_count += 1
            except Exception as e:
                Logger.error(f"Failed to preload cache entry: {e}")
        
        Logger.info(f"Preloaded {loaded_count}/{len(query_response_pairs)} cache entries")
        return loaded_count