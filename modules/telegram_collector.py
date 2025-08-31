"""
Telegram message collector for gathering messages from channels.
"""

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from telethon import TelegramClient
from telethon.tl.types import Message, MessageMediaDocument, MessageMediaPhoto

from .models import ChannelConfig, RateLimitConfig
from .database_service import TelegramDBService

# Import config constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLLECTIONS_DIR, DEFAULT_EXPORT_PATH, F_PREFIX, F_SEPARATOR, F_EXTENSION

logger = logging.getLogger(__name__)


def convert_messages_to_dataframe_format(messages: List[Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Convert TelegramMessage objects to a DataFrame-compatible structured format."""
    if not messages:
        return []
    
    structured_messages = []
    for i, msg in enumerate(messages):
        try:
            if hasattr(msg, '__dict__'):
                # Handle TelegramMessage objects
                msg_dict = {}
                for attr_name, attr_value in msg.__dict__.items():
                    if hasattr(attr_value, 'isoformat'):
                        msg_dict[attr_name] = attr_value.isoformat()
                    else:
                        msg_dict[attr_name] = attr_value
                structured_messages.append(msg_dict)
            elif isinstance(msg, dict):
                # Handle already-dict messages
                msg_dict = {}
                for key, value in msg.items():
                    if hasattr(value, 'isoformat'):
                        msg_dict[key] = value.isoformat()
                    else:
                        msg_dict[key] = value
                structured_messages.append(msg_dict)
            else:
                logger.warning(f"Message {i+1} has unexpected format: {type(msg)}")
                continue
        except Exception as e:
            logger.error(f"Error processing message {i+1}: {e}")
            continue
    
    if structured_messages:
        logger.info(f"🔧 Converted {len(structured_messages)} messages to structured format")
        sample_fields = list(structured_messages[0].keys())
        logger.info(f"📋 Available fields: {', '.join(sample_fields[:10])}{'...' if len(sample_fields) > 10 else ''}")
    
    return structured_messages


def export_messages_to_file(messages: List[Any], export_path: str, channel_list: List[Any], logger: logging.Logger) -> Optional[str]:
    """Export messages to structured file (JSON with DataFrame-like structure)."""
    try:
        # Ensure export directory exists
        os.makedirs(COLLECTIONS_DIR, exist_ok=True)
        
        # Generate filename
        if export_path == DEFAULT_EXPORT_PATH:
            # Create descriptive filename with channel name and message ID range
            # Get channel names (remove @ symbol for filename)
            channel_names = [ch.username.replace('@', '') for ch in channel_list if ch.enabled]
            channel_str = F_SEPARATOR.join(channel_names) if channel_names else 'unknown'
            
            # Get message ID range
            if messages:
                message_ids = [msg.message_id if hasattr(msg, 'message_id') else 
                             (msg.get('message_id') if isinstance(msg, dict) else None) 
                             for msg in messages]
                message_ids = [mid for mid in message_ids if mid is not None]
                if message_ids:
                    min_id = min(message_ids)
                    max_id = max(message_ids)
                    filename = f"{COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}{min_id}{F_SEPARATOR}{max_id}{F_EXTENSION}"
                else:
                    filename = f"{COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}unknown{F_SEPARATOR}unknown{F_EXTENSION}"
            else:
                filename = f"{COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}none{F_SEPARATOR}none{F_EXTENSION}"
        else:
            filename = export_path if export_path.endswith(F_EXTENSION) else f"{export_path}{F_EXTENSION}"
        
        # Convert messages to structured DataFrame-like format
        structured_messages = convert_messages_to_dataframe_format(messages, logger)
        
        # Prepare export data with structured format
        export_data = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'channels': [ch.username for ch in channel_list if ch.enabled],
                'total_messages': len(messages),
                'data_format': 'structured_dataframe',
                'fields': list(structured_messages[0].keys()) if structured_messages else []
            },
            'messages': structured_messages
        }
        
        # Write file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✅ Exported {len(messages)} messages to structured format: {filename}")
        logger.info(f"📊 Data converted to DataFrame-compatible structure with {len(structured_messages[0]) if structured_messages else 0} fields")
        return filename
        
    except Exception as e:
        logger.error(f"Failed to export messages: {e}")
        return None


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
        self.collected_messages = []  # Store messages as they're collected

    
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
                                 offset_id: Optional[int] = None) -> List[Dict[str, Any]]:
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
            if offset_id > 0:
                collection_params['max_id'] = offset_id
            
            async for message in self.client.iter_messages(channel_config.username, **collection_params):
                # Rate limiting
                await self._wait_for_rate_limit()
                
                # Process message
                telegram_message = await self.process_message(message, channel_config.username)
                if telegram_message:
                    messages.append(telegram_message)
                    self.collected_messages.append(telegram_message)  # Store in instance variable
                    message_count += 1
                    self.stats['total_messages'] += 1
                
                # Update database immediately to ensure it's saved
                if self.db_service and telegram_message:
                    try:
                        success = await self.db_service.store_message(telegram_message)
                        if success:
                            logger.debug(f"Successfully stored message {telegram_message['message_id']} in database")
                        else:
                            logger.warning(f"Failed to store message {telegram_message['message_id']} in database")
                    except Exception as e:
                        logger.error(f"Error storing message {telegram_message['message_id']} in database: {e}")
                
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
    
    def get_collected_messages(self) -> List[Dict[str, Any]]:
        """Get all messages collected so far."""
        return self.collected_messages.copy()
    
    def clear_collected_messages(self):
        """Clear the collected messages list."""
        self.collected_messages.clear()
    

    
    async def process_message(self, message: Message, channel_username: str) -> Optional[Dict[str, Any]]:
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
            
            # Create message dictionary
            telegram_message = {
                'message_id': message.id,
                'channel_username': channel_username,
                'date': message_date,
                'text': message_text,
                'media_type': media_type,
                'file_name': file_name,
                'file_size': file_size,
                'mime_type': mime_type,
                'duration': None,  # Could be extracted for video/audio
                'width': None,      # Could be extracted for media
                'height': None,     # Could be extracted for media
                'caption': None,    # Could be extracted
                'views': views_count,
                'forwards': forwards_count,
                'replies': replies_count,
                'edit_date': None,  # Could be extracted
                'is_forwarded': False,  # Could be extracted
                'forwarded_from': None,  # Could be extracted
                'forwarded_message_id': None,  # Could be extracted
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
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
