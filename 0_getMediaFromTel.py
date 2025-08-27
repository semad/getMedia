#!/usr/bin/env python3
"""
Enhanced Telegram Media Collector with Rate Limiting and Database Storage

This script collects media files from multiple Telegram channels with intelligent
rate limiting to avoid hitting API limits. It stores message metadata in a database
and downloads media files to local storage.
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import Message, MessageMediaDocument, MessageMediaPhoto
import aiohttp
import click

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChannelConfig:
    """Configuration for a Telegram channel."""
    username: str
    enabled: bool = True
    max_messages_per_session: int = 100
    download_media: bool = True
    priority: int = 1  # Higher number = higher priority

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    messages_per_minute: int = 30
    media_downloads_per_minute: int = 10
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
    caption: Optional[str]
    views: Optional[int]
    forwards: Optional[int]
    replies: Optional[int]
    is_forwarded: bool
    forwarded_from: Optional[str]
    creator_username: Optional[str]
    creator_first_name: Optional[str]
    creator_last_name: Optional[str]
    created_at: datetime
    updated_at: datetime

class RateLimiter:
    """Rate limiter for Telegram API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.message_timestamps: List[datetime] = []
        self.media_timestamps: List[datetime] = []
        self.last_channel_switch = datetime.now()
    
    async def wait_for_message_limit(self):
        """Wait if we've hit the message rate limit."""
        now = datetime.now()
        # Remove timestamps older than 1 minute
        self.message_timestamps = [ts for ts in self.message_timestamps 
                                 if now - ts < timedelta(minutes=1)]
        
        if len(self.message_timestamps) >= self.config.messages_per_minute:
            # Use a shorter bedtime (10 seconds) instead of waiting the full minute
            wait_time = 10
            logger.info(f"Rate limit reached. Waiting {wait_time} seconds (shortened bedtime)...")
            await asyncio.sleep(wait_time)
            # Clear the timestamps to allow immediate continuation
            self.message_timestamps.clear()
        
        self.message_timestamps.append(now)
    
    async def wait_for_media_limit(self):
        """Wait if we've hit the media download rate limit."""
        now = datetime.now()
        self.media_timestamps = [ts for ts in self.media_timestamps 
                               if now - ts < timedelta(minutes=1)]
        
        if len(self.media_timestamps) >= self.config.media_downloads_per_minute:
            # Use a shorter bedtime (10 seconds) instead of waiting the full minute
            wait_time = 10
            logger.info(f"Media rate limit reached. Waiting {wait_time} seconds (shortened bedtime)...")
            await asyncio.sleep(wait_time)
            # Clear the timestamps to allow immediate continuation
            self.media_timestamps.clear()
        
        self.media_timestamps.append(now)
    
    async def wait_for_channel_switch(self):
        """Wait between channel switches."""
        now = datetime.now()
        time_since_switch = (now - self.last_channel_switch).seconds
        
        if time_since_switch < self.config.delay_between_channels:
            wait_time = self.config.delay_between_channels - time_since_switch
            logger.info(f"Waiting {wait_time} seconds before switching channels...")
            await asyncio.sleep(wait_time)
        
        self.last_channel_switch = now

class TelegramCollector:
    """Enhanced Telegram media collector with rate limiting."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.client: Optional[TelegramClient] = None
        self.db_service: Optional['TelegramDBService'] = None
        self.stats = {
            'total_messages': 0,
            'media_downloads': 0,
            'errors': 0,
            'channels_processed': 0
        }
    
    async def initialize(self, api_id: str, api_hash: str, session_name: str):
        """Initialize the Telegram client."""
        try:
            self.client = TelegramClient(session_name, api_id, api_hash)
            await self.client.start()
            logger.info("Telegram client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            return False
    
    async def collect_from_channel(self, channel_config: ChannelConfig, 
                                 max_messages: Optional[int] = None, 
                                 offset_id: Optional[int] = None) -> List[TelegramMessage]:
        """Collect messages from a specific channel in the order Telegram provides them."""
        if not self.client:
            logger.error("Telegram client not initialized")
            return []
        
        messages = []
        message_count = 0
        # Only use default if max_messages is explicitly 0 or not provided
        if max_messages is None:
            max_messages = None  # Keep as None for unlimited
        elif max_messages == 0:
            max_messages = channel_config.max_messages_per_session
        
        try:
            logger.info(f"Starting collection from channel: {channel_config.username}")
            
            # Build message collection parameters - let Telegram decide the order
            collection_params = {}
            if max_messages is not None:
                collection_params['limit'] = max_messages
            
            # Check if we want older messages (max_id) or newer messages (default)
            if hasattr(self, 'get_older_messages') and self.get_older_messages:
                # Get messages with ID less than the last collected message ID
                if hasattr(self, 'before_id') and self.before_id:
                    # Use manually specified before_id
                    collection_params['max_id'] = self.before_id
                    logger.info(f"Collecting {max_messages or 'all'} messages with ID < {self.before_id} (manually specified)")
                else:
                    # Use the last message ID from database
                    if offset_id and offset_id > 0:
                        collection_params['max_id'] = offset_id
                        logger.info(f"Collecting {max_messages or 'all'} messages with ID < {offset_id} (from database)")
                    else:
                        logger.info(f"No offset ID available, collecting from beginning")
                        collection_params['reverse'] = True
            else:
                # Default: start from the most recent messages (Telegram's natural order)
                logger.info(f"Collecting {max_messages or 'all'} messages in Telegram's natural order")
            
            async for message in self.client.iter_messages(channel_config.username, **collection_params):
                # Rate limiting
                await self.rate_limiter.wait_for_message_limit()
                
                # Process message
                telegram_message = await self.process_message(message, channel_config.username)
                if telegram_message:
                    messages.append(telegram_message)
                    message_count += 1
                    
                    # Print message details if requested
                    if hasattr(self, 'print_messages') and self.print_messages:
                        self.print_message_details(telegram_message)
                
                # Download media if enabled
                if channel_config.download_media and telegram_message.media_type:
                    await self.download_media(message, telegram_message)
                
                # Update database
                if self.db_service:
                    await self.db_service.store_message(telegram_message)
                
                # Progress update
                if message_count % 10 == 0:
                    logger.info(f"Processed {message_count} messages from {channel_config.username} (last message_id: {message.id})")
                
                if max_messages is not None and message_count >= max_messages:
                    break
            
            if message_count > 0:
                logger.info(f"Completed collection from {channel_config.username}: {message_count} messages (last message_id: {messages[-1].message_id if messages else 'N/A'})")
            else:
                logger.info(f"Completed collection from {channel_config.username}: {message_count} messages")
            self.stats['channels_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error collecting from channel {channel_config.username}: {e}")
            self.stats['errors'] += 1
        
        return messages
    
    async def process_message(self, message: Message, channel_username: str) -> Optional[TelegramMessage]:
        """Process a Telegram message and extract metadata."""
        try:
            # Validate basic message data
            if not hasattr(message, 'id') or message.id is None:
                logger.warning(f"Message missing ID, skipping")
                return None
                
            if not hasattr(message, 'date') or message.date is None:
                logger.warning(f"Message {message.id} missing date, using current time")
                message_date = datetime.now()
            else:
                message_date = message.date
            
            # Extract media information safely
            media_type = None
            file_name = None
            file_size = None
            mime_type = None
            
            if hasattr(message, 'media') and message.media:
                try:
                    if isinstance(message.media, MessageMediaDocument):
                        media_type = "document"
                        
                        # Look for file_name in document attributes
                        if hasattr(message.media.document, 'attributes') and message.media.document.attributes:
                            for attr in message.media.document.attributes:
                                if hasattr(attr, 'file_name') and attr.file_name:
                                    file_name = attr.file_name
                                    break
                        else:
                            # No attributes available, try to get filename from document itself
                            file_name = getattr(message.media.document, 'file_name', None)
                        
                        # If no file_name found, try to determine type from mime_type
                        if not file_name and hasattr(message.media.document, 'mime_type'):
                            mime_type = message.media.document.mime_type
                            if mime_type and mime_type.startswith('video/'):
                                media_type = "video"
                            elif mime_type and mime_type.startswith('audio/'):
                                media_type = "audio"
                            elif mime_type and mime_type.startswith('image/'):
                                media_type = "image"
                            elif mime_type and mime_type.startswith('application/'):
                                media_type = "document"
                            elif mime_type and mime_type.startswith('text/'):
                                media_type = "text"
                            else:
                                media_type = "document"  # Default fallback
                        
                        file_size = getattr(message.media.document, 'size', None)
                        mime_type = getattr(message.media.document, 'mime_type', None)
                        
                    elif isinstance(message.media, MessageMediaPhoto):
                        media_type = "photo"
                        file_size = 0  # Photos don't have size in attributes
                        
                except Exception as media_error:
                    logger.warning(f"Error extracting media info from message {message.id}: {media_error}")
                    logger.debug(f"Media object type: {type(message.media)}")
                    if hasattr(message.media, 'document'):
                        logger.debug(f"Document attributes: {[type(attr) for attr in message.media.document.attributes] if hasattr(message.media.document, 'attributes') else 'No attributes'}")
                    media_type = "unknown"
            
            # Extract text safely
            message_text = ""
            if hasattr(message, 'text') and message.text:
                message_text = str(message.text)
            
            # Ensure text meets minimum length requirement (API schema requires min_length=1)
            if not message_text or len(message_text.strip()) == 0:
                message_text = "[No text content]"  # Provide default text for empty messages
            
            # Truncate text if it's too long (API schema max_length not specified, but be safe)
            if message_text and len(message_text) > 10000:  # Reasonable limit
                message_text = message_text[:9997] + "..."  # Leave room for "..."
                logger.warning(f"Text for message {message.id} truncated to 10000 characters")
            
            # Extract replies count safely
            replies_count = None
            try:
                if hasattr(message, 'replies') and message.replies:
                    replies_count = getattr(message.replies, 'replies', None)
            except Exception:
                replies_count = None
            
            # Extract other fields safely to avoid serialization issues
            views_count = None
            forwards_count = None
            try:
                views_count = getattr(message, 'views', None)
                forwards_count = getattr(message, 'forwards', None)
            except Exception:
                views_count = None
                forwards_count = None
            
            # Extract creator information safely
            creator_username = None
            creator_first_name = None
            creator_last_name = None
            
            try:
                if hasattr(message, 'sender') and message.sender:
                    sender = message.sender
                    creator_username = getattr(sender, 'username', None)
                    creator_first_name = getattr(sender, 'first_name', None)
                    creator_last_name = getattr(sender, 'last_name', None)
            except Exception as sender_error:
                logger.warning(f"Error extracting sender info from message {message.id}: {sender_error}")
            
            # Extract caption safely
            caption = None
            try:
                if hasattr(message, 'raw_text') and message.raw_text:
                    caption = str(message.raw_text)
                    # Truncate caption if it's too long (API schema max_length=2000)
                    if caption and len(caption) > 2000:
                        caption = caption[:1997] + "..."  # Leave room for "..."
                        logger.warning(f"Caption for message {message.id} truncated to 2000 characters")
            except Exception:
                caption = None
            
            # Extract forwarding information safely
            is_forwarded = False
            forwarded_from = None
            try:
                if hasattr(message, 'forward') and message.forward:
                    is_forwarded = True
                    if hasattr(message.forward, 'chat') and message.forward.chat:
                        forwarded_from = getattr(message.forward.chat, 'username', None)
            except Exception as forward_error:
                logger.warning(f"Error extracting forward info from message {message.id}: {forward_error}")
            
            telegram_message = TelegramMessage(
                message_id=message.id,
                channel_username=channel_username,
                date=message_date,
                text=message_text,
                media_type=media_type,
                file_name=file_name,
                file_size=file_size,
                mime_type=mime_type,
                caption=caption,
                views=views_count,
                forwards=forwards_count,
                replies=replies_count,
                is_forwarded=is_forwarded,
                forwarded_from=forwarded_from,
                creator_username=creator_username,
                creator_first_name=creator_first_name,
                creator_last_name=creator_last_name,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.stats['total_messages'] += 1
            return telegram_message
            
        except Exception as e:
            logger.error(f"Error processing message {getattr(message, 'id', 'unknown')}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def download_media(self, message: Message, telegram_message: TelegramMessage):
        """Download media from a message."""
        if not telegram_message.media_type or not telegram_message.file_name:
            return
        
        try:
            # Rate limiting for media downloads
            await self.rate_limiter.wait_for_media_limit()
            
            # Create download directory
            download_dir = Path(f"../_downloads/{telegram_message.channel_username}")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            file_path = download_dir / telegram_message.file_name
            if not file_path.exists():
                await message.download_media(str(file_path))
                logger.info(f"Downloaded media: {telegram_message.file_name}")
                self.stats['media_downloads'] += 1
            else:
                logger.info(f"Media already exists: {telegram_message.file_name}")
                
        except Exception as e:
            logger.error(f"Error downloading media from message {message.id}: {e}")
            self.stats['errors'] += 1
    
    async def collect_from_multiple_channels(self, channels: List[ChannelConfig]):
        """Collect from multiple channels with rate limiting."""
        # Sort channels by priority (higher priority first)
        sorted_channels = sorted(channels, key=lambda x: x.priority, reverse=True)
        
        for channel in sorted_channels:
            if not channel.enabled:
                continue
            
            logger.info(f"Processing channel: {channel.username}")
            await self.collect_from_channel(channel)
            
            # Wait between channels
            if channel != sorted_channels[-1]:  # Don't wait after the last channel
                await self.rate_limiter.wait_for_channel_switch()
    
    async def close(self):
        """Close the Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")
    
    def save_messages_to_json(self, messages: List[TelegramMessage], filename: str):
        """Save collected messages to a JSON file."""
        try:
            # Convert messages to serializable format
            json_data = []
            for message in messages:
                message_dict = asdict(message)
                
                # Convert datetime objects to ISO format strings
                if message_dict['date']:
                    message_dict['date'] = message_dict['date'].isoformat()
                if message_dict['created_at']:
                    message_dict['created_at'] = message_dict['created_at'].isoformat()
                if message_dict['updated_at']:
                    message_dict['updated_at'] = message_dict['updated_at'].isoformat()
                
                # Ensure all values are JSON serializable
                for key, value in message_dict.items():
                    if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                        message_dict[key] = str(value)
                
                json_data.append(message_dict)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(messages)} messages to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving messages to JSON: {e}")
    
    def print_message_details(self, message: TelegramMessage):
        """Print detailed message information to console."""
        print("\n" + "="*80)
        print(f"📱 MESSAGE ID: {message.message_id}")
        print(f"📢 CHANNEL: {message.channel_username}")
        
        # Display creator information
        if message.creator_username or message.creator_first_name:
            print(f"👤 CREATOR: ", end="")
            if message.creator_username:
                print(f"@{message.creator_username}", end="")
            if message.creator_first_name:
                if message.creator_username:
                    print(" (", end="")
                print(f"{message.creator_first_name}", end="")
                if message.creator_last_name:
                    print(f" {message.creator_last_name}", end="")
                if message.creator_username:
                    print(")", end="")
            print()  # New line
        
        print(f"📅 DATE: {message.date}")
        print(f"📝 TEXT: {message.text[:100]}{'...' if len(message.text) > 100 else ''}")
        
        if message.media_type:
            print(f"📁 MEDIA TYPE: {message.media_type}")
            if message.file_name:
                print(f"📄 FILE NAME: {message.file_name}")
            if message.file_size:
                print(f"📊 FILE SIZE: {message.file_size:,} bytes")
            if message.mime_type:
                print(f"🔧 MIME TYPE: {message.mime_type}")
        
        if message.caption:
            print(f"💬 CAPTION: {message.caption[:100]}{'...' if len(message.caption) > 100 else ''}")
        
        # Engagement metrics
        if message.views is not None:
            print(f"👁️ VIEWS: {message.views}")
        if message.forwards is not None:
            print(f"🔄 FORWARDS: {message.forwards}")
        if message.replies is not None:
            print(f"💬 REPLIES: {message.replies}")
        
        # Forwarding info
        if message.is_forwarded:
            print(f"🔄 FORWARDED: Yes")
            if message.forwarded_from:
                print(f"📤 FROM: {message.forwarded_from}")
        
        print(f"⏰ CREATED: {message.created_at}")
        print("="*80)

class DatabaseChecker:
    """Check database for existing messages."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_last_message_id(self, channel_username: str) -> Optional[int]:
        """Get the highest message ID from database for a specific channel."""
        if not self.session:
            logger.error("Database session not initialized")
            return None
            
        try:
            url = f"{self.db_url}/api/v1/telegram/messages"
            params = {"channel_username": channel_username, "limit": 1000}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Handle both old list format and new paginated format
                    if isinstance(data, dict) and 'data' in data:
                        # New paginated format: {"data": [...], "total_count": X}
                        messages_list = data.get('data', [])
                    elif isinstance(data, list):
                        # Old list format (fallback)
                        messages_list = data
                    else:
                        logger.warning(f"Unexpected response format: {type(data)}")
                        return None
                    
                    if messages_list and isinstance(messages_list, list):
                        # Find the highest message_id
                        message_ids = [msg.get('message_id') for msg in messages_list if msg.get('message_id') is not None]
                        if message_ids:
                            return max(message_ids)
                    return None
                else:
                    logger.warning(f"Failed to get messages: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting last message ID: {e}")
            return None
    
    async def get_lowest_message_id(self, channel_username: str) -> Optional[int]:
        """Get the lowest message ID from database for a specific channel."""
        if not self.session:
            logger.error("Database session not initialized")
            return None
            
        try:
            url = f"{self.db_url}/api/v1/telegram/messages"
            params = {"channel_username": channel_username, "limit": 1000}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Handle both old list format and new paginated format
                    if isinstance(data, dict) and 'data' in data:
                        # New paginated format: {"data": [...], "total_count": X}
                        messages_list = data.get('data', [])
                    elif isinstance(data, list):
                        # Old list format (fallback)
                        messages_list = data
                    else:
                        logger.warning(f"Unexpected response format: {type(data)}")
                        return None
                    
                    if messages_list and isinstance(messages_list, list):
                        # Find the lowest message_id
                        message_ids = [msg.get('message_id') for msg in messages_list if msg.get('message_id') is not None]
                        if message_ids:
                            return min(message_ids)
                    return None
                else:
                    logger.warning(f"Failed to get messages: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting lowest message ID: {e}")
            return None
    
    async def get_database_stats(self) -> dict:
        """Get database statistics."""
        if not self.session:
            logger.error("Database session not initialized")
            return {}
            
        try:
            url = f"{self.db_url}/api/v1/telegram/stats"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get stats: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


class TelegramDBService:
    """Service for storing Telegram message metadata in database."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def store_message(self, message: TelegramMessage):
        """Store a Telegram message in the database."""
        try:
            # Convert to dict for API
            message_data = asdict(message)
            
            # Remove fields that are not in the API schema
            fields_to_remove = ['created_at', 'updated_at']
            for field in fields_to_remove:
                message_data.pop(field, None)
            
            # Safely convert datetime objects to ISO format strings
            if message_data['date']:
                message_data['date'] = message_data['date'].isoformat()
            
            # Ensure all values are JSON serializable
            for key, value in message_data.items():
                if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                    message_data[key] = str(value)
            
            # Send to microservice
            url = f"{self.db_url}/api/v1/telegram/messages"
            async with self.session.post(url, json=message_data) as response:
                if response.status == 200:  # API returns 200, not 201
                    logger.debug(f"Stored message {message.message_id} in database")
                else:
                    logger.warning(f"Failed to store message {message.message_id}: {response.status}")
                    # Log response body for debugging
                    try:
                        error_body = await response.text()
                        logger.warning(f"Error details for message {message.message_id}: {error_body}")
                        logger.warning(f"Message data that failed: {message_data}")
                    except:
                        pass
                    
        except Exception as e:
            logger.error(f"Error storing message {message.message_id}: {e}")
            logger.debug(f"Message data that failed: {message_data if 'message_data' in locals() else 'Not available'}")

@click.command(help='Enhanced Telegram Media Collector with Rate Limiting')
@click.option('--channels', '-c', 
              help='Comma-separated list of channel usernames')
@click.option('--max-messages', '-m', default=None, type=int,
              help='Maximum messages to collect per channel (default: no limit)')
@click.option('--offset-id', '-o', default=0, type=int,
              help='Start collecting from message ID greater than this (default: 0 for complete history)')
@click.option('--download-media', is_flag=True, default=False,
              help='Download media files')
@click.option('--no-download-media', is_flag=True, default=False,
              help='Disable media downloads')
@click.option('--save-json', default=None, metavar='FILENAME',
              help='Save collected messages to JSON file. Use --save-json filename.json or --save-json for default name')
@click.option('--save-json-default', is_flag=True, default=False,
              help='Save collected messages to JSON file with default filename (telegram_messages.json)')
@click.option('--print-messages', is_flag=True, default=False,
              help='Print full message details to console')
@click.option('--rate-limit', '-r', default=120, type=int,
              help='Messages per minute rate limit (default: 120)')
@click.option('--session-name', '-s', default='telegram_collector',
              help='Telegram session name')
@click.option('--dry-run', is_flag=True,
              help='Run without downloading or storing')
@click.option('--help', '-h', is_flag=True, help='Show this help message and exit')
@click.option('--version', is_flag=True, help='Show version information and exit')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--older', is_flag=True, help='Collect older messages instead of newer ones')
@click.option('--before-id', type=int, help='Collect messages with ID less than this value (for getting previous batches)')
def main(channels: str, max_messages: int, offset_id: int, download_media: bool, no_download_media: bool,
         save_json: str, save_json_default: bool, print_messages: bool, rate_limit: int, session_name: str, 
         dry_run: bool, help: bool, version: bool, verbose: bool, older: bool, before_id: int):
    """Enhanced Telegram Media Collector with Rate Limiting."""
    
    # Show help if requested
    if help:
        click.echo(click.get_current_context().get_help())
        return
    
    # Show version if requested
    if version:
        click.echo("Enhanced Telegram Media Collector v1.0.0")
        click.echo("Built with Telethon and Click")
        sys.exit(0)
    
    # Configure verbose logging if requested
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Load configuration
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    
    if not api_id or not api_hash:
        logger.error("Missing TG_API_ID or TG_API_HASH environment variables")
        return
    
    # Parse channels
    if channels:
        channel_list = [ChannelConfig(username=ch.strip()) for ch in channels.split(',')]
    else:
        # Default channels
        channel_list = [
            ChannelConfig("@SherwinVakiliLibrary", priority=1),
            # Add more channels here
        ]
    
    # Handle media download settings
    if no_download_media:
        download_media = False
    
    # Set media download settings for all channels
    for channel in channel_list:
        channel.download_media = download_media
    
    # Configure rate limiting
    rate_config = RateLimitConfig(
        messages_per_minute=rate_limit,
        media_downloads_per_minute=rate_limit // 3,  # More conservative for media
        delay_between_channels=5,
        session_cooldown=300
    )
    
    async def run_collector():
        collector = TelegramCollector(rate_config)
        # Set the print_messages flag
        collector.print_messages = print_messages
        # Set the get_older_messages flag
        collector.get_older_messages = older
        # Set the before_id parameter
        collector.before_id = before_id
        all_messages = []
        
        # Check database for message IDs to determine where to continue from
        actual_offset_id = offset_id
        if offset_id == 0:  # Default value, check database
            logger.info("Checking database for message IDs...")
            async with DatabaseChecker(db_url) as checker:
                for channel in channel_list:
                    if channel.enabled:
                        if older:  # For older mode, we need the lowest message ID
                            lowest_id = await checker.get_lowest_message_id(channel.username)
                            if lowest_id:
                                actual_offset_id = lowest_id
                                logger.info(f"Found lowest message ID in database: {lowest_id} (for older collection)")
                            else:
                                logger.info(f"No existing messages found for {channel.username}, starting from beginning")
                                actual_offset_id = 0
                        else:  # For newer mode, we need the highest message ID
                            highest_id = await checker.get_last_message_id(channel.username)
                            if highest_id:
                                actual_offset_id = highest_id
                                logger.info(f"Found highest message ID in database: {highest_id} (for newer collection)")
                            else:
                                logger.info(f"No existing messages found for {channel.username}, starting from beginning")
                                actual_offset_id = 0
        
        # Initialize database service
        if not dry_run:
            collector.db_service = TelegramDBService(db_url)
            async with collector.db_service:
                # Initialize Telegram client
                if await collector.initialize(api_id, api_hash, session_name):
                    # Collect from channels
                    for channel in channel_list:
                        if channel.enabled:
                            messages = await collector.collect_from_channel(channel, max_messages, actual_offset_id)
                            all_messages.extend(messages)
                    await collector.close()
        else:
            logger.info("DRY RUN MODE - No downloads or database operations")
            if await collector.initialize(api_id, api_hash, session_name):
                for channel in channel_list:
                    if channel.enabled:
                        messages = await collector.collect_from_channel(channel, max_messages, actual_offset_id)
                        all_messages.extend(messages)
                await collector.close()
        
        # Save to JSON if requested
        should_save_json = save_json or save_json_default
        if should_save_json and all_messages:
            # Determine filename: use provided name or default
            if save_json:
                filename = save_json
            else:
                filename = 'telegram_messages.json'
            collector.save_messages_to_json(all_messages, filename)
        
        # Print statistics
        logger.info("Collection completed!")
        logger.info(f"Total messages processed: {collector.stats['total_messages']}")
        logger.info(f"Media downloads: {collector.stats['media_downloads']}")
        logger.info(f"Channels processed: {collector.stats['channels_processed']}")
        logger.info(f"Errors encountered: {collector.stats['errors']}")
        if should_save_json:
            if save_json:
                filename = save_json
            else:
                filename = 'telegram_messages.json'
            logger.info(f"Messages saved to: {filename}")
    
    # Run the collector
    asyncio.run(run_collector())

if __name__ == '__main__':
    main()
