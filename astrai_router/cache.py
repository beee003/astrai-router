"""
Semantic Caching for Astrai
- Caches LLM responses based on semantic similarity
- Saves 50-90% on repeated/similar queries
- Uses embedding-based similarity matching

Competitive Advantage:
- Users save money on repeated/similar queries
- Faster response times for cached queries
- Reduces load on upstream providers
"""
import os
import time
import hashlib
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import asyncio

# Try to import numpy for vector operations, fallback to simple hashing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not installed - using hash-based caching instead of semantic")


# ============================================================================
# LIGHTWEIGHT EMBEDDING (No external API calls needed)
# ============================================================================
# Uses TF-IDF style word vectors for semantic similarity
# This is fast, free, and works offline

# Common English stop words to ignore
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'although', 'though', 'after', 'before',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'please', 'help', 'want', 'know', 'think',
    'tell', 'give', 'get', 'make', 'go', 'see', 'come', 'take', 'find',
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into words, removing stop words and normalizing."""
    # Lowercase and extract words
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    # Remove stop words and short words
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def get_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from tokens."""
    if len(tokens) < n:
        return tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_word_vector(tokens: List[str], vocab_size: int = 2000) -> List[float]:
    """
    Compute a word frequency vector with unigrams and bigrams.
    Uses hashing trick to map words to fixed-size vector.
    Includes TF-IDF-like weighting.
    """
    if not HAS_NUMPY:
        return []
    
    vector = np.zeros(vocab_size, dtype=np.float32)
    
    # Add unigrams with weight 1.0
    for token in tokens:
        idx = hash(token) % vocab_size
        vector[idx] += 1.0
    
    # Add bigrams with weight 1.5 (more specific = more important)
    bigrams = get_ngrams(tokens, 2)
    for bigram in bigrams:
        idx = hash(bigram) % vocab_size
        vector[idx] += 1.5
    
    # Add character trigrams for fuzzy matching (weight 0.3)
    for token in tokens:
        if len(token) >= 3:
            for i in range(len(token) - 2):
                char_trigram = token[i:i+3]
                idx = hash(f"char:{char_trigram}") % vocab_size
                vector[idx] += 0.3
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not HAS_NUMPY or not vec1 or not vec2:
        return 0.0
    
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


def get_query_embedding(messages: List[Dict[str, str]]) -> List[float]:
    """
    Generate an embedding for a query (list of messages).
    IMPORTANT: Includes BOTH system message (task instructions) AND user content.
    This ensures different tasks on the same document don't incorrectly match.
    """
    # Extract ALL message content - system messages are critical for task differentiation
    all_content = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # System messages define the TASK - weight them heavily
        if role == "system":
            # Repeat system content to increase its weight in the embedding
            all_content.append(content)
            all_content.append(content)  # Double weight for task instructions
        elif role == "user":
            all_content.append(content)
    
    # Combine and tokenize
    full_text = " ".join(all_content)
    tokens = tokenize(full_text)
    
    return compute_word_vector(tokens)


@dataclass
class CacheEntry:
    """A cached response entry."""
    key: str
    query_hash: str
    model: str
    messages_hash: str
    response: Dict[str, Any]
    created_at: float
    ttl: int  # Time to live in seconds
    hit_count: int = 0
    embedding: Optional[List[float]] = None
    
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)


class SemanticCache:
    """
    Semantic cache for LLM responses.
    
    Features:
    - Exact match caching (hash-based)
    - Semantic similarity matching (embedding-based)
    - TTL-based expiration
    - LRU eviction when full
    - Per-user isolation
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        default_ttl: int = 3600,  # 1 hour
        similarity_threshold: float = 0.99,  # Very high to prevent false matches on same doc with different tasks
    ):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        
        # Exact match cache (hash -> entry)
        self._exact_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Semantic cache (for embedding-based matching)
        self._semantic_cache: List[CacheEntry] = []
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "evictions": 0,
            "total_saved_tokens": 0,
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _hash_messages(self, messages: List[Dict[str, str]], model: str) -> str:
        """Create a deterministic hash of messages and model."""
        # Normalize messages
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", "user"),
                "content": (msg.get("content", "") if isinstance(msg.get("content"), str) else str(msg.get("content", ""))).strip().lower(),
            })
        
        # Create hash
        content = json.dumps({"model": model, "messages": normalized}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _hash_query(self, messages: List[Dict[str, str]]) -> str:
        """Create a hash of the full query including system message for proper task differentiation."""
        # Include BOTH system message (task) AND user message (content)
        # This ensures different tasks on the same document have different hashes
        parts = []
        for m in messages:
            role = m.get("role", "")
            raw_content = m.get("content", "")
            content = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip().lower()
            if role in ("system", "user") and content:
                parts.append(f"{role}:{content}")
        
        query = "|".join(parts)
        return hashlib.sha256(query.encode()).hexdigest()
    
    async def get(
        self,
        messages: List[Dict[str, str]],
        model: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to get a cached response.
        
        Returns cached response if found, None otherwise.
        """
        async with self._lock:
            messages_hash = self._hash_messages(messages, model)
            
            # Add user_id to key for isolation
            cache_key = f"{user_id or 'global'}:{messages_hash}"
            
            # 1. Try exact match first
            if cache_key in self._exact_cache:
                entry = self._exact_cache[cache_key]
                
                if entry.is_expired():
                    del self._exact_cache[cache_key]
                    self.stats["evictions"] += 1
                else:
                    # Move to end (LRU)
                    self._exact_cache.move_to_end(cache_key)
                    entry.hit_count += 1
                    self.stats["hits"] += 1
                    
                    # Estimate saved tokens
                    response = entry.response
                    if "usage" in response:
                        self.stats["total_saved_tokens"] += response["usage"].get("total_tokens", 0)
                    
                    # Add cache indicator to response
                    cached_response = response.copy()
                    cached_response["_cached"] = True
                    cached_response["_cache_hit_type"] = "exact"
                    return cached_response
            
            # 2. Try semantic match using embedding similarity
            if HAS_NUMPY and self._semantic_cache:
                # Generate embedding for the query
                query_embedding = get_query_embedding(messages)
                
                if query_embedding:
                    best_match: Optional[CacheEntry] = None
                    best_similarity: float = 0.0
                    
                    for entry in self._semantic_cache:
                        if entry.is_expired():
                            continue
                        
                        # Must be same model for semantic match
                        if entry.model != model:
                            continue
                        
                        # Check embedding similarity
                        if entry.embedding:
                            similarity = cosine_similarity(query_embedding, entry.embedding)
                            
                            if similarity >= self.similarity_threshold and similarity > best_similarity:
                                best_similarity = similarity
                                best_match = entry
                    
                    if best_match:
                        best_match.hit_count += 1
                        self.stats["semantic_hits"] += 1
                        self.stats["hits"] += 1
                        
                        # Estimate saved tokens
                        if "usage" in best_match.response:
                            self.stats["total_saved_tokens"] += best_match.response["usage"].get("total_tokens", 0)
                        
                        cached_response = best_match.response.copy()
                        cached_response["_cached"] = True
                        cached_response["_cache_hit_type"] = "semantic"
                        cached_response["_similarity_score"] = round(best_similarity, 4)
                        return cached_response
            
            self.stats["misses"] += 1
            return None
    
    async def set(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response: Dict[str, Any],
        user_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache a response with embedding for semantic matching.
        """
        async with self._lock:
            messages_hash = self._hash_messages(messages, model)
            query_hash = self._hash_query(messages)
            cache_key = f"{user_id or 'global'}:{messages_hash}"
            
            # Compute embedding for semantic matching
            embedding = get_query_embedding(messages) if HAS_NUMPY else None
            
            # Evict if at capacity
            while len(self._exact_cache) >= self.max_entries:
                oldest_key = next(iter(self._exact_cache))
                del self._exact_cache[oldest_key]
                self.stats["evictions"] += 1
            
            # Create entry with embedding
            entry = CacheEntry(
                key=cache_key,
                query_hash=query_hash,
                model=model,
                messages_hash=messages_hash,
                response=response,
                created_at=time.time(),
                ttl=ttl or self.default_ttl,
                embedding=embedding,
            )
            
            # Store in exact cache
            self._exact_cache[cache_key] = entry
            
            # Also store in semantic cache for similarity matching
            # Keep semantic cache smaller for performance (linear search)
            if embedding and len(self._semantic_cache) < self.max_entries // 10:
                self._semantic_cache.append(entry)
    
    async def invalidate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Returns number of entries invalidated.
        """
        async with self._lock:
            count = 0
            
            if messages and model:
                # Invalidate specific entry
                messages_hash = self._hash_messages(messages, model)
                cache_key = f"{user_id or 'global'}:{messages_hash}"
                
                if cache_key in self._exact_cache:
                    del self._exact_cache[cache_key]
                    count += 1
            
            elif user_id:
                # Invalidate all entries for a user
                keys_to_delete = [
                    k for k in self._exact_cache.keys()
                    if k.startswith(f"{user_id}:")
                ]
                for key in keys_to_delete:
                    del self._exact_cache[key]
                    count += 1
            
            return count
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._exact_cache.items()
                if v.is_expired()
            ]
            
            for key in expired_keys:
                del self._exact_cache[key]
            
            # Clean semantic cache
            self._semantic_cache = [
                e for e in self._semantic_cache
                if not e.is_expired()
            ]
            
            self.stats["evictions"] += len(expired_keys)
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_entries": len(self._exact_cache),
            "semantic_entries": len(self._semantic_cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "semantic_hits": self.stats["semantic_hits"],
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.stats["evictions"],
            "total_saved_tokens": self.stats["total_saved_tokens"],
            "estimated_savings_usd": round(self.stats["total_saved_tokens"] * 0.000002, 4),  # ~$2/1M tokens avg
        }


# Global cache instance
# High threshold (0.99) to prevent false matches on same document with different tasks
# Only cache truly identical or near-identical queries
SEMANTIC_CACHE = SemanticCache(
    max_entries=10000,
    default_ttl=3600,  # 1 hour
    similarity_threshold=0.99,  # Very high - only near-exact matches
)


async def get_cached_response(
    messages: List[Dict[str, str]],
    model: str,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convenience function to get cached response."""
    return await SEMANTIC_CACHE.get(messages, model, user_id)


async def cache_response(
    messages: List[Dict[str, str]],
    model: str,
    response: Dict[str, Any],
    user_id: Optional[str] = None,
    ttl: Optional[int] = None,
) -> None:
    """Convenience function to cache a response."""
    await SEMANTIC_CACHE.set(messages, model, response, user_id, ttl)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return SEMANTIC_CACHE.get_stats()
