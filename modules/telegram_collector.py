"""
Telegram message collector for gathering messages from channels.
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from telethon import TelegramClient
from telethon.tl.types import Message, MessageMediaDocument, MessageMediaPhoto

from .models import TelegramMessage, ChannelConfig, RateLimitConfig
from .database_service import TelegramDBService

logger = logging.getLogger(__name__)


class TelegramCollector:
    """Collects messages from Telegram channels."""
    
    def __init__(self, rate_config: RateLimitConfig):
        self.client: Optional[TelegramClient] = None
        self.db_service: Optional[TelegramDBService] = None
        self.rate_config = rate_config
        self.stats = {
            'total_messages': 0,
            'channels_processed': 0,
            'errors': 0
        }
    
    async def initialize(self, api_id: str, api_hash: str, session_name: str) -> bool:
        """Initialize the Telegram client."""
        try:
            self.client = TelegramClient(session_name, int(api_id), api_hash)
            await self.client.start()
            logger.info("Telegram client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            return False
    
    async def collect_from_channel(self, channel_config: ChannelConfig, 
                                 max_messages: Optional[int] = None, 
                                 offset_id: Optional[int] = None) -> List[TelegramMessage]:
        """Collect messages from a specific channel."""
        if not self.client:
            logger.error("Telegram client not initialized")
            return []
        
        messages = []
        message_count = 0
        
        if max_messages is None:
            max_messages = None
        elif max_messages == 0:
            max_messages = channel_config.max_messages_per_session
        
        try:
            logger.info(f"Starting collection from channel: {channel_config.username}")
            
            collection_params = {}
            if max_messages is not None:
                collection_params['limit'] = max_messages
            
            async for message in self.client.iter_messages(channel_config.username, **collection_params):
                # Rate limiting
                await self._wait_for_rate_limit()
                
                # Process message
                telegram_message = await self.process_message(message, channel_config.username)
                if telegram_message:
                    messages.append(telegram_message)
                    message_count += 1
                    self.stats['total_messages'] += 1
                
                # Update database
                if self.db_service:
                    await self.db_service.store_message(telegram_message)
                
                # Progress update
                if message_count % 10 == 0:
                    logger.info(f"Processed {message_count} messages from {channel_config.username}")
                
                if max_messages is not None and message_count >= max_messages:
                    break
            
            logger.info(f"Completed collection from {channel_config.username}: {message_count} messages")
            self.stats['channels_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error collecting from channel {channel_config.username}: {e}")
            self.stats['errors'] += 1
        
        return messages
    
    async def process_message(self, message: Message, channel_username: str) -> Optional[TelegramMessage]:
        """Process a Telegram message and extract metadata."""
        try:
            if not hasattr(message, 'id') or message.id is None:
                return None
                
            if not hasattr(message, 'date') or message.date is None:
                message_date = datetime.now()
            else:
                message_date = message.date
            
            # Extract media information
            media_type = None
            file_name = None
            file_size = None
            mime_type = None
            
            if hasattr(message, 'media') and message.media:
                if isinstance(message.media, MessageMediaDocument):
                    media_type = "document"
                    if hasattr(message.media.document, 'attributes') and message.media.document.attributes:
                        for attr in message.media.document.attributes:
                            if hasattr(attr, 'file_name') and attr.file_name:
                                file_name = attr.file_name
                                break
                    file_size = getattr(message.media.document, 'size', None)
                    mime_type = getattr(message.media.document, 'mime_type', None)
                    
                elif isinstance(message.media, MessageMediaPhoto):
                    media_type = "photo"
                    file_size = 0
            
            # Extract text
            message_text = ""
            if hasattr(message, 'text') and message.text:
                message_text = str(message.text)
            
            if not message_text or len(message_text.strip()) == 0:
                message_text = "[No text content]"
            
            # Extract other fields
            replies_count = None
            views_count = None
            forwards_count = None
            
            if hasattr(message, 'replies') and message.replies:
                replies_count = message.replies.replies
            
            if hasattr(message, 'views'):
                views_count = message.views
            
            if hasattr(message, 'forwards'):
                forwards_count = message.forwards
            
            # Create TelegramMessage object
            telegram_message = TelegramMessage(
                message_id=message.id,
                channel_username=channel_username,
                date=message_date,
                text=message_text,
                media_type=media_type,
                file_name=file_name,
                file_size=file_size,
                mime_type=mime_type,
                duration=None,  # Could be extracted for video/audio
                width=None,      # Could be extracted for media
                height=None,     # Could be extracted for media
                caption=None,    # Could be extracted
                views=views_count,
                forwards=forwards_count,
                replies=replies_count,
                edit_date=None,  # Could be extracted
                is_forwarded=False,  # Could be extracted
                forwarded_from=None,  # Could be extracted
                forwarded_message_id=None,  # Could be extracted
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return telegram_message
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    async def _wait_for_rate_limit(self):
        """Wait for rate limiting."""
        await asyncio.sleep(60 / self.rate_config.messages_per_minute)
    
    async def close(self):
        """Close the Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")


class DatabaseChecker:
    """Utility class for checking database state."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_last_message_id(self, channel_username: str) -> Optional[int]:
        """Get the highest message ID for a channel from the database."""
        # This would need to be implemented based on your database schema
        # For now, return None to indicate no existing messages
        return None
