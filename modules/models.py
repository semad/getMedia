"""
Data models for Telegram Media Messages Tool.
"""

from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Configuration for a Telegram channel."""

    username: str
    enabled: bool = True
    max_messages_per_session: int = 100
    priority: int = 1  # Higher number = higher priority


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    messages_per_minute: int = 30
    delay_between_channels: int = 5  # seconds
    session_cooldown: int = 300  # 5 minutes between sessions
