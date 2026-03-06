"""
Input Validation Module

Production-grade input validation for API endpoints.
Prevents common security issues and ensures data integrity.
"""

import re
import ipaddress
import socket
from typing import Optional, List, Tuple, Any
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, code: str = "VALIDATION_ERROR"):
        self.field = field
        self.message = message
        self.code = code
        super().__init__(f"{field}: {message}")


class ErrorCode(str, Enum):
    """Standard error codes for API responses."""
    INVALID_URL = "INVALID_URL"
    INVALID_PROTOCOL = "INVALID_PROTOCOL"
    INVALID_LENGTH = "INVALID_LENGTH"
    INVALID_FORMAT = "INVALID_FORMAT"
    INVALID_RANGE = "INVALID_RANGE"
    REQUIRED_FIELD = "REQUIRED_FIELD"
    INVALID_MODEL = "INVALID_MODEL"
    INVALID_PROVIDER = "INVALID_PROVIDER"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"


# =============================================================================
# WEBHOOK VALIDATION
# =============================================================================

# Allowed webhook URL patterns
ALLOWED_WEBHOOK_HOSTS = [
    r".*\.slack\.com$",
    r".*\.discord\.com$",
    r".*\.zapier\.com$",
    r".*\.pipedream\.net$",
    r"hooks\..*$",
    r"api\..*$",
    r"webhook\..*$",
    r".*\.ngrok\.io$",
    r".*\.ngrok-free\.app$",
]

# Blocked patterns (security)
BLOCKED_URL_PATTERNS = [
    r"^(https?://)?localhost",
    r"^(https?://)?127\.",
    r"^(https?://)?0\.",
    r"^(https?://)?10\.",
    r"^(https?://)?172\.(1[6-9]|2[0-9]|3[0-1])\.",
    r"^(https?://)?192\.168\.",
    r"^file://",
    r"^javascript:",
    r"^data:",
    r"^ftp://",
]

# Valid webhook events
VALID_WEBHOOK_EVENTS = {
    "budget_exceeded",
    "budget_warning",
    "model_down",
    "model_recovered",
    "quality_alert",
    "latency_spike",
    "rate_limit_hit",
    "fallback_triggered",
    "daily_summary",
    "weekly_report",
}


def validate_webhook_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate webhook URL for security and format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is required"

    if len(url) > 2048:
        return False, "URL exceeds maximum length of 2048 characters"

    # Check for blocked patterns (SSRF prevention)
    for pattern in BLOCKED_URL_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return False, "URL points to a restricted address"

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    # Require HTTPS
    if parsed.scheme != "https":
        return False, "URL must use HTTPS protocol"

    # Require valid host
    if not parsed.netloc:
        return False, "URL must have a valid host"

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL must have a valid host"

    # Block localhost variations
    if host in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
        return False, "localhost URLs are not allowed"

    def _is_restricted_ip(ip) -> bool:
        # Only allow globally routable targets.
        return not ip.is_global

    def _is_private_host(h: str) -> bool:
        try:
            ip = ipaddress.ip_address(h)
            return _is_restricted_ip(ip)
        except ValueError:
            # hostname
            if h.endswith(".local"):
                return True
            return False

    if _is_private_host(host):
        return False, "URL points to a restricted address"

    # Resolve DNS and reject hostnames that map to restricted addresses.
    try:
        addr_info = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False, "Host could not be resolved"
    except Exception:
        return False, "Failed to resolve host"

    resolved_ips = set()
    for _, _, _, _, sockaddr in addr_info:
        try:
            resolved_ips.add(ipaddress.ip_address(sockaddr[0]))
        except (ValueError, IndexError):
            continue

    if not resolved_ips:
        return False, "Host could not be resolved"

    if any(_is_restricted_ip(ip) for ip in resolved_ips):
        return False, "URL resolves to a restricted address"

    return True, None


def validate_webhook_events(events: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate webhook event types.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not events:
        return False, "At least one event type is required"

    if len(events) > 20:
        return False, "Maximum 20 event types allowed"

    invalid_events = [e for e in events if e not in VALID_WEBHOOK_EVENTS]
    if invalid_events:
        return False, f"Invalid event types: {', '.join(invalid_events)}"

    return True, None


def validate_webhook_secret(secret: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate webhook secret for signing.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if secret is None:
        return True, None  # Secret is optional

    if len(secret) < 16:
        return False, "Secret must be at least 16 characters"

    if len(secret) > 255:
        return False, "Secret exceeds maximum length of 255 characters"

    return True, None


def validate_webhook_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate webhook name.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Name is required"

    if len(name) > 100:
        return False, "Name exceeds maximum length of 100 characters"

    # Allow alphanumeric, spaces, hyphens, underscores
    if not re.match(r'^[\w\s\-]+$', name):
        return False, "Name contains invalid characters"

    return True, None


# =============================================================================
# FALLBACK CHAIN VALIDATION
# =============================================================================

# Known models (can be extended)
KNOWN_MODELS = {
    # Anthropic
    "claude-4-opus", "claude-4-sonnet", "claude-3.5-sonnet", "claude-3-opus",
    "claude-3-sonnet", "claude-3-haiku",
    # OpenAI
    "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "o1", "o1-preview", "o1-mini", "o3-mini",
    # Google
    "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
    # DeepSeek
    "deepseek-v3.2", "deepseek-v3", "deepseek-r1", "deepseek-coder",
    # Meta
    "llama-3.3-70b", "llama-3.3-405b", "llama-3.2-70b",
    # Mistral
    "mistral-large", "mixtral-8x22b", "mixtral-8x7b",
}

KNOWN_PROVIDERS = {
    "anthropic", "openai", "google", "deepseek", "together", "groq",
    "fireworks", "azure", "bedrock", "vertex", "mistral",
}


def validate_model(model: str) -> Tuple[bool, Optional[str]]:
    """
    Validate model identifier.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model:
        return False, "Model is required"

    if len(model) > 100:
        return False, "Model name exceeds maximum length"

    # Allow unknown models but warn
    # In production, you might want to be stricter
    return True, None


def validate_provider(provider: str) -> Tuple[bool, Optional[str]]:
    """
    Validate provider identifier.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not provider:
        return False, "Provider is required"

    if len(provider) > 50:
        return False, "Provider name exceeds maximum length"

    if provider.lower() not in KNOWN_PROVIDERS:
        return False, f"Unknown provider: {provider}"

    return True, None


def validate_fallback_chain_step(step: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate a single fallback chain step.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(step, dict):
        return False, "Step must be an object"

    # Validate model
    model = step.get('model')
    is_valid, error = validate_model(model)
    if not is_valid:
        return False, f"model: {error}"

    # Validate provider
    provider = step.get('provider')
    is_valid, error = validate_provider(provider)
    if not is_valid:
        return False, f"provider: {error}"

    # Validate max_latency_ms (optional)
    max_latency = step.get('max_latency_ms')
    if max_latency is not None:
        if not isinstance(max_latency, int) or max_latency < 100 or max_latency > 60000:
            return False, "max_latency_ms must be between 100 and 60000"

    # Validate max_cost_usd (optional)
    max_cost = step.get('max_cost_usd')
    if max_cost is not None:
        if not isinstance(max_cost, (int, float)) or max_cost < 0 or max_cost > 100:
            return False, "max_cost_usd must be between 0 and 100"

    return True, None


def validate_fallback_chain(chain: List[dict]) -> Tuple[bool, Optional[str]]:
    """
    Validate entire fallback chain.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not chain:
        return False, "Chain must have at least one step"

    if len(chain) > 10:
        return False, "Chain cannot have more than 10 steps"

    for i, step in enumerate(chain):
        is_valid, error = validate_fallback_chain_step(step)
        if not is_valid:
            return False, f"Step {i + 1}: {error}"

    return True, None


def validate_chain_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate fallback chain name.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Name is required"

    if len(name) > 100:
        return False, "Name exceeds maximum length of 100 characters"

    return True, None


def validate_chain_settings(settings: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate fallback chain settings.

    Returns:
        Tuple of (is_valid, error_message)
    """
    max_retries = settings.get('max_retries', 3)
    if not isinstance(max_retries, int) or max_retries < 1 or max_retries > 10:
        return False, "max_retries must be between 1 and 10"

    total_timeout = settings.get('total_timeout_ms', 30000)
    if not isinstance(total_timeout, int) or total_timeout < 1000 or total_timeout > 120000:
        return False, "total_timeout_ms must be between 1000 and 120000"

    min_quality = settings.get('min_quality_score', 0.7)
    if not isinstance(min_quality, (int, float)) or min_quality < 0 or min_quality > 1:
        return False, "min_quality_score must be between 0 and 1"

    return True, None


# =============================================================================
# GENERAL VALIDATION
# =============================================================================

def validate_quality_score(score: float) -> Tuple[bool, Optional[str]]:
    """
    Validate quality score.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(score, (int, float)):
        return False, "Score must be a number"

    if score < 0 or score > 5:
        return False, "Score must be between 0 and 5"

    return True, None


def validate_task_type(task_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate task type.

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_task_types = {
        "code", "research", "chat", "creative", "analysis",
        "math", "translation", "summarization", "general",
    }

    if not task_type:
        return True, None  # Optional

    if task_type.lower() not in valid_task_types:
        return False, f"Invalid task type: {task_type}"

    return True, None


def validate_user_id(user_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user ID format (UUID).

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not user_id:
        return False, "User ID is required"

    # UUID format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, user_id, re.IGNORECASE):
        return False, "Invalid user ID format"

    return True, None


# =============================================================================
# RATE LIMITING
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000


class RateLimiter:
    """
    Simple in-memory rate limiter.

    In production, use Redis or similar for distributed rate limiting.
    """

    def __init__(self):
        self._requests: dict = {}  # {key: [(timestamp, count), ...]}

    def is_allowed(
        self,
        key: str,
        config: RateLimitConfig = RateLimitConfig()
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        import time
        now = time.time()

        if key not in self._requests:
            self._requests[key] = []

        # Clean up old entries
        self._requests[key] = [
            (ts, count) for ts, count in self._requests[key]
            if now - ts < 86400  # Keep last 24 hours
        ]

        # Count requests in each window
        minute_count = sum(
            count for ts, count in self._requests[key]
            if now - ts < 60
        )
        hour_count = sum(
            count for ts, count in self._requests[key]
            if now - ts < 3600
        )
        day_count = sum(
            count for ts, count in self._requests[key]
            if now - ts < 86400
        )

        # Check limits
        if minute_count >= config.requests_per_minute:
            return False, f"Rate limit exceeded: {config.requests_per_minute}/minute"

        if hour_count >= config.requests_per_hour:
            return False, f"Rate limit exceeded: {config.requests_per_hour}/hour"

        if day_count >= config.requests_per_day:
            return False, f"Rate limit exceeded: {config.requests_per_day}/day"

        # Record request
        self._requests[key].append((now, 1))

        return True, None


# Singleton rate limiter
RATE_LIMITER = RateLimiter()


# =============================================================================
# WEBHOOK RATE LIMITING
# =============================================================================

WEBHOOK_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=10,   # Max 10 webhook deliveries per minute
    requests_per_hour=100,    # Max 100 per hour
    requests_per_day=1000,    # Max 1000 per day
)


def check_webhook_rate_limit(webhook_id: str) -> Tuple[bool, Optional[str]]:
    """
    Check if webhook delivery is allowed.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    return RATE_LIMITER.is_allowed(f"webhook:{webhook_id}", WEBHOOK_RATE_LIMIT)
