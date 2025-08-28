"""
Data models for Telegram Media Messages Tool.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any


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


@dataclass
class TelegramMessage:
    """Telegram message metadata."""
    message_id: int
    channel_username: str
    date: datetime
    text: str
    media_type: Optional[str]
    file_name: Optional[str]
    file_size: Optional[int]
    mime_type: Optional[str]
    duration: Optional[int]
    width: Optional[int]
    height: Optional[int]
    caption: Optional[str]
    views: Optional[int]
    forwards: Optional[int]
    replies: Optional[int]
    edit_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_forwarded: bool = False
    forwarded_from: Optional[str] = None
    forwarded_message_id: Optional[int] = None


@dataclass
class ImportStats:
    """Statistics for import operations."""
    total_messages: int = 0
    imported_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_messages == 0:
            return 0.0
        return (self.imported_count / self.total_messages) * 100
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate total duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
