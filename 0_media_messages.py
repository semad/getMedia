#!/usr/bin/env python3
"""
Unified Telegram Media Messages Tool

This script provides five main modes:
1. collect - Collect messages from Telegram channels
2. export - Export existing data from database
3. analyze - Analyze collected data with console reports and interactive HTML dashboards
4. analyze-docname - Analyze PDF/EPUB document filenames for uniqueness and duplicates
5. import - Import data from JSON/CSV files into database via API

Usage:
    ./0_media_messages.py collect [OPTIONS]        # Collect new messages
    ./0_media_messages.py export [OPTIONS]         # Export existing data
    ./0_media_messages.py analyze [OPTIONS]        # Analyze data with optional dashboard
    ./0_media_messages.py analyze-docname [OPTIONS] # Analyze document filenames
    ./0_media_messages.py import [OPTIONS]         # Import data to database
"""

import os
import sys
import time
import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from collections import Counter, defaultdict

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import Message, MessageMediaDocument, MessageMediaPhoto
import aiohttp
import click
import pandas as pd

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

# ============================================================================
# DATA MODELS
# ============================================================================

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

# ============================================================================
# RETRY HANDLER
# ============================================================================

class RetryHandler:
    """Handles retries with exponential backoff for failed operations."""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_count = 0
    
    async def execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} after {delay:.1f}s delay...")
                    await asyncio.sleep(delay)
                
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation succeeded on retry attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    self.retry_count += 1
                else:
                    logger.error(f"Operation failed after {self.max_retries} retries: {e}")
                    raise last_exception
    
    def reset(self):
        """Reset retry counter."""
        self.retry_count = 0

# ============================================================================
# COLLECTION MODE
# ============================================================================

class RateLimiter:
    """Rate limiter for Telegram API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.message_timestamps: List[datetime] = []
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
                await self.rate_limiter.wait_for_message_limit()
                
                # Process message
                telegram_message = await self.process_message(message, channel_config.username)
                if telegram_message:
                    messages.append(telegram_message)
                    message_count += 1
                
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
            
            try:
                if hasattr(message, 'replies') and message.replies:
                    replies_count = getattr(message.replies, 'replies', None)
                views_count = getattr(message, 'views', None)
                forwards_count = getattr(message, 'forwards', None)
            except Exception:
                pass
            
            # Extract creator information
            creator_username = None
            creator_first_name = None
            creator_last_name = None
            
            try:
                if hasattr(message, 'sender') and message.sender:
                    sender = message.sender
                    creator_username = getattr(sender, 'username', None)
                    creator_first_name = getattr(sender, 'first_name', None)
                    creator_last_name = getattr(sender, 'last_name', None)
            except Exception:
                pass
            
            # Extract caption
            caption = None
            try:
                if hasattr(message, 'raw_text') and message.raw_text:
                    caption = str(message.raw_text)
            except Exception:
                pass
            
            # Extract forwarding information
            is_forwarded = False
            forwarded_from = None
            try:
                if hasattr(message, 'forward') and message.forward:
                    is_forwarded = True
                    if hasattr(message.forward, 'chat') and message.forward.chat:
                        forwarded_from = getattr(message.forward.chat, 'username', None)
            except Exception:
                pass
            
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
    
    async def close(self):
        """Close the Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")

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
            return None
            
        try:
            url = f"{self.db_url}/api/v1/telegram/messages"
            params = {"channel_username": channel_username, "limit": 1000}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and 'data' in data:
                        messages_list = data.get('data', [])
                    elif isinstance(data, list):
                        messages_list = data
                    else:
                        return None
                    
                    if messages_list and isinstance(messages_list, list):
                        message_ids = [msg.get('message_id') for msg in messages_list if msg.get('message_id') is not None]
                        if message_ids:
                            return max(message_ids)
                    return None
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting last message ID: {e}")
            return None
    
    async def get_database_stats(self) -> dict:
        """Get database statistics."""
        if not self.session:
            return {}
            
        try:
            url = f"{self.db_url}/api/v1/telegram/stats"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
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
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def store_message(self, message: TelegramMessage):
        """Store a Telegram message in the database."""
        try:
            message_data = asdict(message)
            
            # Remove fields that are not in the API schema
            fields_to_remove = ['created_at', 'updated_at']
            for field in fields_to_remove:
                message_data.pop(field, None)
            
            # Convert datetime objects to ISO format strings
            if message_data['date']:
                message_data['date'] = message_data['date'].isoformat()
            
            # Clean the message data to ensure JSON serialization compatibility
            message_data = self.clean_message_data(message_data)
            
            # Send to microservice
            url = f"{self.db_url}/api/v1/telegram/messages"
            if self.session:
                async with self.session.post(url, json=message_data) as response:
                    if response.status == 200:
                        logger.debug(f"Stored message {message.message_id} in database")
                        return True
                    else:
                        logger.warning(f"Failed to store message {message.message_id}: {response.status}")
                        return False
            else:
                logger.error("No active session for database service")
                return False
                    
        except Exception as e:
            logger.error(f"Error storing message {message.message_id}: {e}")
            return False
    
    async def check_connection(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            url = f"{self.db_url}/api/v1/telegram/stats"
            if self.session:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
            return False
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
            return False
    
    def clean_message_data(self, message_data: dict) -> dict:
        """Clean message data to ensure JSON serialization compatibility."""
        cleaned_data: dict = {}
        
        for key, value in message_data.items():
            if value is None:
                cleaned_data[key] = None
                continue
                
            if isinstance(value, float):
                # Handle problematic float values
                if math.isnan(value) or math.isinf(value):
                    cleaned_data[key] = None
                else:
                    cleaned_data[key] = value
            elif isinstance(value, datetime):
                # Convert datetime objects to ISO format strings
                cleaned_data[key] = value.isoformat()
            elif isinstance(value, timedelta):
                # Convert timedelta to total seconds
                cleaned_data[key] = value.total_seconds()
            elif isinstance(value, (str, int, bool, list, dict)):
                # These types are already JSON serializable
                cleaned_data[key] = value
            else:
                # Convert other types to strings
                try:
                    cleaned_data[key] = str(value)
                except Exception:
                    # If conversion fails, set to None
                    cleaned_data[key] = None
                    logger.warning(f"Could not serialize field {key} with value {value}, setting to None")
        
        return cleaned_data

# ============================================================================
# EXPORT MODE
# ============================================================================

class TelegramMessageExporter:
    """Export Telegram messages from database using pandas."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_total_count(self) -> int:
        """Get total number of messages in database."""
        try:
            url = f"{self.db_url}/api/v1/telegram/stats"
            async with self.session.get(url) as response:
                if response.status == 200:
                    stats = await response.json()
                    return stats.get('total_messages', 0)
                else:
                    return 0
        except Exception as e:
            logger.error(f"Error getting total count: {e}")
            return 0
    
    async def get_all_messages(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Get all messages from database in batches."""
        all_messages = []
        page = 1
        
        try:
            while True:
                logger.info(f"Fetching batch {page} (batch size: {batch_size})...")
                
                url = f"{self.db_url}/api/v1/telegram/messages"
                params = {
                    "page": page,
                    "items_per_page": batch_size
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, dict) and 'data' in data:
                            messages = data.get('data', [])
                            total_count = data.get('total_count', 0)
                        else:
                            messages = data if isinstance(data, list) else []
                            total_count = len(messages)
                        
                        if not messages:
                            break
                        
                        all_messages.extend(messages)
                        logger.info(f"Fetched {len(messages)} messages (total so far: {len(all_messages)}/{total_count})")
                        
                        if len(all_messages) >= total_count:
                            break
                        
                        page += 1
                    else:
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
        
        return all_messages
    
    def create_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert messages to pandas DataFrame with data cleaning and optimization."""
        if not messages:
            return pd.DataFrame()
        
        df = pd.DataFrame(messages)
        
        # Data cleaning and optimization
        logger.info("Processing and cleaning data...")
        
        # Convert date columns to datetime
        date_columns = ['date', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle numeric columns
        numeric_columns = ['message_id', 'views', 'forwards', 'replies', 'file_size']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['text', 'caption', 'file_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Add derived columns for analysis
        df['text_length'] = df['text'].str.len()
        df['has_media'] = df['media_type'].notna() & (df['media_type'] != '')
        df['is_forwarded'] = df['is_forwarded'].fillna(False)
        
        # Reorder columns for better readability
        priority_columns = ['message_id', 'channel_username', 'date', 'text', 'media_type', 'file_name']
        other_columns = [col for col in df.columns if col not in priority_columns]
        df = df[priority_columns + other_columns]
        
        logger.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def export_to_json(self, df: pd.DataFrame, filename: str, format_type: str = "pretty") -> bool:
        """Export DataFrame to JSON format."""
        try:
            export_data = {
                "export_info": {
                    "exported_at": datetime.now().isoformat(),
                    "total_messages": len(df),
                    "format": "pandas_dataframe",
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict()
                },
                "messages": df.to_dict('records')
            }
            
            if format_type == "pretty":
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, default=str)
            
            logger.info(f"Data exported to JSON: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON {filename}: {e}")
            return False
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to CSV format."""
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Data exported to CSV: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV {filename}: {e}")
            return False
    
    def export_to_excel(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to Excel format with multiple sheets."""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Messages', index=False)
                
                # Summary statistics sheet
                summary_data = {
                    'Metric': [
                        'Total Messages',
                        'Unique Channels',
                        'Messages with Media',
                        'Forwarded Messages',
                        'Date Range Start',
                        'Date Range End',
                        'Average Text Length',
                        'Most Common Media Type'
                    ],
                    'Value': [
                        len(df),
                        df['channel_username'].nunique(),
                        df['has_media'].sum(),
                        df['is_forwarded'].sum(),
                        df['date'].min() if 'date' in df.columns else 'N/A',
                        df['date'].max() if 'date' in df.columns else 'N/A',
                        df['text_length'].mean() if 'text_length' in df.columns else 'N/A',
                        df['media_type'].mode().iloc[0] if 'media_type' in df.columns and not df['media_type'].empty else 'N/A'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Channel statistics sheet
                if 'channel_username' in df.columns:
                    channel_stats = df.groupby('channel_username').agg({
                        'message_id': 'count',
                        'has_media': 'sum',
                        'is_forwarded': 'sum',
                        'text_length': 'mean'
                    }).rename(columns={
                        'message_id': 'message_count',
                        'has_media': 'media_count',
                        'is_forwarded': 'forwarded_count',
                        'text_length': 'avg_text_length'
                    })
                    channel_stats.to_excel(writer, sheet_name='Channel Statistics')
                
                # Media type analysis sheet
                if 'media_type' in df.columns:
                    media_stats = df['media_type'].value_counts().reset_index()
                    media_stats.columns = ['Media Type', 'Count']
                    media_stats.to_excel(writer, sheet_name='Media Analysis', index=False)
            
            logger.info(f"Data exported to Excel: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Excel {filename}: {e}")
            return False
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive summary report of the data."""
        if df.empty:
            return "No data to analyze"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TELEGRAM MESSAGES DATA SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("BASIC STATISTICS:")
        report_lines.append(f"  Total Messages: {len(df):,}")
        report_lines.append(f"  Unique Channels: {df['channel_username'].nunique()}")
        report_lines.append(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
        report_lines.append("")
        
        # Channel breakdown
        report_lines.append("CHANNEL BREAKDOWN:")
        channel_counts = df['channel_username'].value_counts()
        for channel, count in channel_counts.head(10).items():
            report_lines.append(f"  {channel}: {count:,} messages")
        if len(channel_counts) > 10:
            report_lines.append(f"  ... and {len(channel_counts) - 10} more channels")
        report_lines.append("")
        
        # Media analysis
        if 'media_type' in df.columns:
            report_lines.append("MEDIA ANALYSIS:")
            media_counts = df['media_type'].value_counts()
            for media_type, count in media_counts.items():
                if pd.notna(media_type) and media_type:
                    report_lines.append(f"  {media_type}: {count:,} messages")
            report_lines.append(f"  No Media: {df['media_type'].isna().sum():,} messages")
            report_lines.append("")
        
        # Text analysis
        if 'text_length' in df.columns:
            report_lines.append("TEXT ANALYSIS:")
            report_lines.append(f"  Average Text Length: {df['text_length'].mean():.1f} characters")
            report_lines.append(f"  Shortest Message: {df['text_length'].min()} characters")
            report_lines.append(f"  Longest Message: {df['text_length'].max()} characters")
            report_lines.append("")
        
        # Forwarding analysis
        if 'is_forwarded' in df.columns:
            report_lines.append("FORWARDING ANALYSIS:")
            forwarded_count = df['is_forwarded'].sum()
            report_lines.append(f"  Forwarded Messages: {forwarded_count:,} ({forwarded_count/len(df)*100:.1f}%)")
            report_lines.append(f"  Original Messages: {len(df) - forwarded_count:,} ({(len(df) - forwarded_count)/len(df)*100:.1f}%)")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

# ============================================================================
# ANALYSIS MODE
# ============================================================================

class DocumentNameAnalyzer:
    """Analyze PDF and EPUB document filenames for uniqueness and duplicates."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare data from file."""
        try:
            logger.info(f"Loading data from {self.data_file}")
            
            if self.data_file.endswith('.json'):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle the structured JSON format with export_info and data
                if isinstance(data, dict) and 'export_info' in data:
                    # Extract the actual data array
                    if 'data' in data:
                        data_array = data['data']
                    else:
                        # If no data key, try to find the messages array
                        for key, value in data.items():
                            if key != 'export_info' and isinstance(value, list):
                                data_array = value
                                break
                        else:
                            raise ValueError("Could not find data array in JSON file")
                    
                    logger.info(f"Found structured JSON with {len(data_array)} messages")
                    self.df = pd.DataFrame(data_array)
                else:
                    # Handle simple array format
                    self.df = pd.DataFrame(data)
                    
            elif self.data_file.endswith('.csv'):
                self.df = pd.read_csv(self.data_file)
            else:
                logger.error(f"Unsupported file format: {self.data_file}")
                return False
            
            # Convert date columns
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Convert numeric columns
            numeric_columns = ['message_id', 'views', 'forwards', 'replies', 'file_size']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_document_names(self):
        """Analyze PDF and EPUB document filenames."""
        logger.info("Analyzing document filenames...")
        
        # Filter for PDF and EPUB documents
        doc_data = self.df[
            (self.df['media_type'] == 'document') & 
            (self.df['file_name'].notna()) &
            (self.df['file_name'] != '')
        ].copy()
        
        if len(doc_data) == 0:
            logger.warning("No documents with filenames found")
            return None
        
        # Filter for PDF and EPUB files
        pdf_epub_data = doc_data[
            doc_data['file_name'].str.lower().str.endswith(('.pdf', '.epub'), na=False)
        ].copy()
        
        if len(pdf_epub_data) == 0:
            logger.warning("No PDF or EPUB files found")
            return None
        
        logger.info(f"Found {len(pdf_epub_data)} PDF/EPUB documents")
        
        # Analyze filenames
        filename_counts = Counter(pdf_epub_data['file_name'])
        
        # Separate unique and non-unique filenames
        unique_filenames = {name: count for name, count in filename_counts.items() if count == 1}
        non_unique_filenames = {name: count for name, count in filename_counts.items() if count > 1}
        
        # Analyze file types
        pdf_files = pdf_epub_data[pdf_epub_data['file_name'].str.lower().str.endswith('.pdf', na=False)]
        epub_files = pdf_epub_data[pdf_epub_data['file_name'].str.lower().str.endswith('.epub', na=False)]
        
        # Create detailed analysis
        analysis = {
            'total_documents': len(pdf_epub_data),
            'pdf_count': len(pdf_files),
            'epub_count': len(epub_files),
            'unique_filenames_count': len(unique_filenames),
            'non_unique_filenames_count': len(non_unique_filenames),
            'total_unique_names': len(set(pdf_epub_data['file_name'])),
            'duplicate_ratio': len(non_unique_filenames) / len(set(pdf_epub_data['file_name'])) * 100,
            'filename_counts': filename_counts,
            'unique_filenames': unique_filenames,
            'non_unique_filenames': non_unique_filenames,
            'pdf_files': pdf_files,
            'epub_files': pdf_epub_data
        }
        
        return analysis
    
    def analyze_duplicates(self, analysis):
        """Analyze duplicate filenames in detail."""
        if not analysis or not analysis['non_unique_filenames']:
            return None
        
        logger.info("Analyzing duplicate filenames...")
        
        duplicates_analysis = {}
        
        for filename, count in analysis['non_unique_filenames'].items():
            # Get all instances of this filename
            instances = self.df[
                (self.df['file_name'] == filename) & 
                (self.df['media_type'] == 'document')
            ].copy()
            
            # Analyze differences between instances
            duplicate_info = {
                'count': count,
                'instances': instances,
                'size_variations': instances['file_size'].unique().tolist() if 'file_size' in instances.columns else [],
                'channel_sources': instances['channel_username'].unique().tolist() if 'channel_username' in instances.columns else [],
                'creator_variations': instances['creator_username'].unique().tolist() if 'creator_username' in instances.columns else [],
                'date_range': {
                    'earliest': instances['date'].min() if 'date' in instances.columns else None,
                    'latest': instances['date'].max() if 'date' in instances.columns else None
                } if 'date' in instances.columns else {},
                'view_variations': instances['views'].unique().tolist() if 'views' in instances.columns else [],
                'forward_variations': instances['forwards'].unique().tolist() if 'forwards' in instances.columns else []
            }
            
            duplicates_analysis[filename] = duplicate_info
        
        return duplicates_analysis
    
    def generate_report(self, analysis, duplicates_analysis):
        """Generate a comprehensive report."""
        if not analysis:
            logger.error("No analysis data available")
            return
        
        logger.info("Generating document analysis report...")
        
        print("\n" + "="*80)
        print("📚 DOCUMENT FILENAME ANALYSIS REPORT")
        print("="*80)
        
        # Summary Statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total PDF/EPUB Documents: {analysis['total_documents']:,}")
        print(f"  PDF Files: {analysis['pdf_count']:,}")
        print(f"  EPUB Files: {analysis['epub_count']:,}")
        print(f"  Unique Filenames: {analysis['unique_filenames_count']:,}")
        print(f"  Non-Unique Filenames: {analysis['non_unique_filenames_count']:,}")
        print(f"  Total Unique Names: {analysis['total_unique_names']:,}")
        print(f"  Duplicate Ratio: {analysis['duplicate_ratio']:.1f}%")
        
        # File Type Breakdown
        print(f"\n📁 FILE TYPE BREAKDOWN:")
        print(f"  PDF Files: {analysis['pdf_count']:,} ({analysis['pdf_count']/analysis['total_documents']*100:.1f}%)")
        print(f"  EPUB Files: {analysis['epub_count']:,} ({analysis['epub_count']/analysis['total_documents']*100:.1f}%)")
        
        # Duplicate Analysis
        if duplicates_analysis:
            print(f"\n🔄 DUPLICATE FILENAME ANALYSIS:")
            print(f"  Files with Duplicate Names: {len(duplicates_analysis)}")
            
            # Show top duplicates
            top_duplicates = sorted(
                [(name, info['count']) for name, info in duplicates_analysis.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            print(f"\n  Top 10 Most Duplicated Filenames:")
            for i, (filename, count) in enumerate(top_duplicates, 1):
                print(f"    {i:2d}. {filename[:60]:<60} (×{count})")
            
            # Analyze duplicate patterns
            print(f"\n  Duplicate Patterns:")
            size_variations = sum(1 for info in duplicates_analysis.values() if len(info['size_variations']) > 1)
            channel_variations = sum(1 for info in duplicates_analysis.values() if len(info['channel_sources']) > 1)
            creator_variations = sum(1 for info in duplicates_analysis.values() if len(info['creator_variations']) > 1)
            
            print(f"    Files with different sizes: {size_variations}")
            print(f"    Files from different channels: {channel_variations}")
            print(f"    Files from different creators: {creator_variations}")
        
        # Unique Filenames Sample
        if analysis['unique_filenames']:
            print(f"\n✅ UNIQUE FILENAMES SAMPLE (showing first 10):")
            unique_sample = list(analysis['unique_filenames'].keys())[:10]
            for i, filename in enumerate(unique_sample, 1):
                print(f"    {i:2d}. {filename}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if analysis['duplicate_ratio'] > 10:
            print(f"  ⚠️  High duplicate ratio ({analysis['duplicate_ratio']:.1f}%) - consider deduplication")
        else:
            print(f"  ✅ Low duplicate ratio ({analysis['duplicate_ratio']:.1f}%) - good filename uniqueness")
        
        if duplicates_analysis:
            print(f"  🔍 Review {len(duplicates_analysis)} files with duplicate names")
            print(f"  📏 Check for files with same names but different sizes (potential quality variations)")
            print(f"  🌐 Verify files from different channels aren't duplicates")
        
        print("\n" + "="*80)
    
    def export_duplicates_csv(self, duplicates_analysis, output_file='duplicate_documents.csv'):
        """Export duplicate analysis to CSV."""
        if not duplicates_analysis:
            logger.warning("No duplicate data to export")
            return
        
        logger.info(f"Exporting duplicate analysis to {output_file}")
        
        # Prepare data for CSV export
        csv_data = []
        
        for filename, info in duplicates_analysis.items():
            for idx, instance in info['instances'].iterrows():
                row = {
                    'filename': filename,
                    'duplicate_count': info['count'],
                    'file_size': instance.get('file_size', 'N/A'),
                    'channel_username': instance.get('channel_username', 'N/A'),
                    'creator_username': instance.get('creator_username', 'N/A'),
                    'date': instance.get('date', 'N/A'),
                    'views': instance.get('views', 'N/A'),
                    'forwards': instance.get('forwards', 'N/A'),
                    'message_id': instance.get('message_id', 'N/A')
                }
                csv_data.append(row)
        
        # Create DataFrame and export
        df_export = pd.DataFrame(csv_data)
        df_export.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Exported {len(csv_data)} duplicate instances to {output_file}")
        
        return output_file

class TelegramDataAnalyzer:
    """Analyze Telegram message data with comprehensive visualization capabilities."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare data from file."""
        try:
            logger.info(f"Loading data from {self.data_file}")
            
            if self.data_file.endswith('.json'):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle the structured JSON format with export_info and data
                if isinstance(data, dict) and 'export_info' in data:
                    # Extract the actual data array
                    if 'data' in data:
                        data_array = data['data']
                    else:
                        # If no data key, try to find the messages array
                        for key, value in data.items():
                            if key != 'export_info' and isinstance(value, list):
                                data_array = value
                                break
                        else:
                            raise ValueError("Could not find data array in JSON file")
                    
                    logger.info(f"Found structured JSON with {len(data_array)} messages")
                    self.df = pd.DataFrame(data_array)
                else:
                    # Handle simple array format
                    self.df = pd.DataFrame(data)
                    
            elif self.data_file.endswith('.csv'):
                self.df = pd.read_csv(self.data_file)
            else:
                logger.error(f"Unsupported file format: {self.data_file}")
                return False
            
            # Convert date columns
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df['year'] = self.df['date'].dt.year
                self.df['month'] = self.df['date'].dt.month
                self.df['day_of_week'] = self.df['date'].dt.day_name()
                self.df['hour'] = self.df['date'].dt.hour
            
            # Convert numeric columns
            numeric_columns = ['message_id', 'views', 'forwards', 'replies', 'file_size']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Handle text length
            if 'text' in self.df.columns:
                self.df['text_length'] = self.df['text'].str.len()
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_media_types(self):
        """Analyze media types distribution."""
        if self.df is None or 'media_type' not in self.df.columns:
            return
        
        print("\n" + "=" * 50)
        print("MEDIA TYPE ANALYSIS")
        print("=" * 50)
        
        media_counts = self.df['media_type'].value_counts()
        for media_type, count in media_counts.items():
            if pd.notna(media_type) and media_type:
                percentage = (count / len(self.df)) * 100
                print(f"{media_type}: {count:,} messages ({percentage:.1f}%)")
        
        no_media = self.df['media_type'].isna().sum()
        no_media_percentage = (no_media / len(self.df)) * 100
        print(f"No Media: {no_media:,} messages ({no_media_percentage:.1f}%)")
    
    def analyze_channels(self):
        """Analyze channel distribution."""
        if self.df is None or 'channel_username' not in self.df.columns:
            return
        
        print("\n" + "=" * 50)
        print("CHANNEL ANALYSIS")
        print("=" * 50)
        
        channel_counts = self.df['channel_username'].value_counts()
        for channel, count in channel_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{channel}: {count:,} messages ({percentage:.1f}%)")
    
    def analyze_text_lengths(self):
        """Analyze text length distribution."""
        if self.df is None or 'text' not in self.df.columns:
            return
        
        print("\n" + "=" * 50)
        print("TEXT LENGTH ANALYSIS")
        print("=" * 50)
        
        text_lengths = self.df['text'].str.len()
        print(f"Average text length: {text_lengths.mean():.1f} characters")
        print(f"Median text length: {text_lengths.median():.1f} characters")
        print(f"Shortest message: {text_lengths.min()} characters")
        print(f"Longest message: {text_lengths.max()} characters")
        
        # Text length distribution
        print("\nText length distribution:")
        bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
        labels = ['0-10', '11-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
        text_lengths_binned = pd.cut(text_lengths, bins=bins, labels=labels, include_lowest=True)
        length_dist = text_lengths_binned.value_counts().sort_index()
        
        for length_range, count in length_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {length_range}: {count:,} messages ({percentage:.1f}%)")
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in messages."""
        if self.df is None or 'date' not in self.df.columns:
            return
        
        print("\n" + "=" * 50)
        print("TEMPORAL PATTERN ANALYSIS")
        print("=" * 50)
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Daily message counts
        daily_counts = self.df['date'].dt.date.value_counts().sort_index()
        print("Top 10 most active days:")
        for date, count in daily_counts.head(10).items():
            print(f"  {date}: {count:,} messages")
        
        # Hourly message counts
        hourly_counts = self.df['date'].dt.hour.value_counts().sort_index()
        print("\nHourly message distribution:")
        for hour, count in hourly_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {hour:02d}:00: {count:,} messages ({percentage:.1f}%)")
    
    def generate_full_report(self):
        """Generate a comprehensive analysis report."""
        if self.df is None:
            print("No data loaded for analysis")
            return
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TELEGRAM DATA ANALYSIS REPORT")
        print("=" * 60)
        print(f"Data source: {self.data_file}")
        print(f"Total messages: {len(self.df):,}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Unique channels: {self.df['channel_username'].nunique()}")
        print("=" * 60)
        
        self.analyze_media_types()
        self.analyze_channels()
        self.analyze_text_lengths()
        self.analyze_temporal_patterns()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
    
    def generate_html_dashboard(self, output_file='telegram_analysis_dashboard.html'):
        """Generate an interactive HTML dashboard with Plotly charts."""
        try:
            # Import plotly here to avoid dependency issues if not installed
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly is required for HTML dashboard generation. Install with: pip install plotly")
            return None
        
        logger.info("Generating HTML dashboard...")
        
        # Create all charts
        charts = {}
        
        # Time series chart
        if 'date' in self.df.columns:
            daily_counts = self.df.groupby(self.df['date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'message_count']
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['message_count'],
                mode='lines+markers',
                name='Messages per Day',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title='Telegram Messages Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Messages',
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            charts['Time Series'] = fig
        
        # Media distribution chart
        if 'media_type' in self.df.columns:
            media_counts = self.df['media_type'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=media_counts.index,
                values=media_counts.values,
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            
            fig.update_layout(
                title='Distribution of Media Types',
                template='plotly_white'
            )
            charts['Media Distribution'] = fig
        
        # Channel comparison chart
        if 'channel_username' in self.df.columns:
            channel_counts = self.df['channel_username'].value_counts()
            fig = go.Figure(data=[go.Bar(
                x=channel_counts.index,
                y=channel_counts.values,
                marker_color='#9467bd'
            )])
            
            fig.update_layout(
                title='Messages by Channel',
                xaxis_title='Channel Username',
                yaxis_title='Number of Messages',
                template='plotly_white'
            )
            charts['Channel Comparison'] = fig
        
        # Text length distribution
        if 'text_length' in self.df.columns:
            fig = go.Figure(data=[go.Histogram(
                x=self.df['text_length'],
                nbinsx=50,
                marker_color='#d62728'
            )])
            
            fig.update_layout(
                title='Distribution of Message Text Lengths',
                xaxis_title='Text Length (characters)',
                yaxis_title='Number of Messages',
                template='plotly_white'
            )
            charts['Text Length Distribution'] = fig
        
        # Summary statistics table
        stats = {
            'Total Messages': len(self.df),
            'Unique Channels': self.df['channel_username'].nunique() if 'channel_username' in self.df.columns else 'N/A',
            'Date Range': f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}" if 'date' in self.df.columns else 'N/A',
            'Messages with Media': self.df['media_type'].notna().sum() if 'media_type' in self.df.columns else 'N/A',
            'Forwarded Messages': self.df['is_forwarded'].sum() if 'is_forwarded' in self.df.columns else 'N/A',
            'Average Text Length': f"{self.df['text_length'].mean():.1f}" if 'text_length' in self.df.columns else 'N/A'
        }
        
        # Add file size statistics if available
        if 'file_size' in self.df.columns:
            file_size_stats = self.df['file_size'].dropna()
            if len(file_size_stats) > 0:
                total_size_mb = file_size_stats.sum() / (1024 * 1024)
                avg_size_mb = file_size_stats.mean() / (1024 * 1024)
                stats.update({
                    'Total File Size': f"{total_size_mb:,.1f} MB",
                    'Average File Size': f"{avg_size_mb:.2f} MB"
                })
        
        # Create summary table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1f77b4',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[[k for k in stats.keys()], [v for v in stats.values()]],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title='Summary Statistics',
            template='plotly_white'
        )
        charts['Summary Statistics'] = fig
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Telegram Data Analysis Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .chart-container {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chart-title {{
                    color: #333;
                    font-size: 1.5em;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }}
                .grid-row {{
                    display: flex;
                    gap: 30px;
                    margin-bottom: 30px;
                    flex-wrap: wrap;
                }}
                .chart-container {{
                    flex: 1;
                    min-width: 500px;
                    max-width: calc(50% - 15px);
                }}
                .full-width {{
                    max-width: 100%;
                }}
                .footer {{
                    text-align: center;
                    color: #666;
                    margin-top: 40px;
                    padding: 20px;
                    border-top: 1px solid #ddd;
                }}
                @media (max-width: 768px) {{
                    .grid-row {{
                        flex-direction: column;
                        gap: 20px;
                    }}
                    .chart-container {{
                        min-width: 100%;
                        max-width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📱 Telegram Data Analysis Dashboard</h1>
                <p>Comprehensive analysis of {len(self.df):,} messages</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add charts in a grid layout
        chart_count = 0
        current_row = []
        
        for title, chart in charts.items():
            if chart is None:
                continue
            
            # Determine if chart should be full width
            is_full_width = title in ['Time Series', 'Summary Statistics']
            
            if is_full_width:
                # Close current row if it has charts
                if current_row:
                    html_content += '<div class="grid-row">'
                    for chart_html in current_row:
                        html_content += chart_html
                    html_content += '</div>'
                    current_row = []
                
                # Add full-width chart
                html_content += f'''
                <div class="chart-container full-width">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
            else:
                # Add chart to current row
                chart_html = f'''
                <div class="chart-container">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
                current_row.append(chart_html)
                
                # If we have 2 charts in the row, close it
                if len(current_row) == 2:
                    html_content += '<div class="grid-row">'
                    for chart_html in current_row:
                        html_content += chart_html
                    html_content += '</div>'
                    current_row = []
        
        # Close any remaining charts in the last row
        if current_row:
            html_content += '<div class="grid-row">'
            for chart_html in current_row:
                html_content += chart_html
            html_content += '</div>'
        
        # Add footer
        html_content += '''
            <div class="footer">
                <p>Dashboard generated using Python, Pandas, and Plotly</p>
                <p>Data source: Telegram Messages Export</p>
            </div>
        </body>
        </html>
        '''
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_file}")
        return output_file

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Unified Telegram Media Messages Tool
    
    This tool provides five main modes:
    1. collect - Collect messages from Telegram channels
    2. export - Export existing data from database
    3. analyze - Analyze collected data with console reports and interactive HTML dashboards
    4. analyze-docname - Analyze PDF/EPUB document filenames for uniqueness and duplicates
    5. import - Import data from JSON/CSV files into database via API
    """
    pass

@cli.command()
@click.option('--channels', '-c', 
              help='Comma-separated list of channel usernames')
@click.option('--max-messages', '-m', default=None, type=int,
              help='Maximum messages to collect per channel (default: no limit)')
@click.option('--offset-id', '-o', default=0, type=int,
              help='Start collecting from message ID greater than this (default: 0 for complete history)')
@click.option('--rate-limit', '-r', default=120, type=int,
              help='Messages per minute rate limit (default: 120)')
@click.option('--session-name', '-s', default='telegram_collector',
              help='Telegram session name')
@click.option('--dry-run', is_flag=True,
              help='Run without storing to database')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging output')
def collect(channels, max_messages, offset_id, rate_limit, session_name, dry_run, verbose):
    """Collect messages from Telegram channels."""
    
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
        channel_list = [
            ChannelConfig("@SherwinVakiliLibrary", priority=1),
        ]
    
    # Configure rate limiting
    rate_config = RateLimitConfig(
        messages_per_minute=rate_limit,
        delay_between_channels=5,
        session_cooldown=300
    )
    
    async def run_collector():
        collector = TelegramCollector(rate_config)
        all_messages = []
        
        # Check database for message IDs
        actual_offset_id = offset_id
        if offset_id == 0:
            logger.info("Checking database for message IDs...")
            async with DatabaseChecker(db_url) as checker:
                for channel in channel_list:
                    if channel.enabled:
                        highest_id = await checker.get_last_message_id(channel.username)
                        if highest_id:
                            actual_offset_id = highest_id
                            logger.info(f"Found highest message ID in database: {highest_id}")
        
        # Initialize database service
        if not dry_run:
            collector.db_service = TelegramDBService(db_url)
            async with collector.db_service:
                if await collector.initialize(api_id, api_hash, session_name):
                    for channel in channel_list:
                        if channel.enabled:
                            messages = await collector.collect_from_channel(channel, max_messages, actual_offset_id)
                            all_messages.extend(messages)
                    await collector.close()
        else:
            logger.info("DRY RUN MODE - No database operations")
            if await collector.initialize(api_id, api_hash, session_name):
                for channel in channel_list:
                    if channel.enabled:
                        messages = await collector.collect_from_channel(channel, max_messages, actual_offset_id)
                        all_messages.extend(messages)
                await collector.close()
        
        # Print statistics
        logger.info("Collection completed!")
        logger.info(f"Total messages processed: {collector.stats['total_messages']}")
        logger.info(f"Channels processed: {collector.stats['channels_processed']}")
        logger.info(f"Errors encountered: {collector.stats['errors']}")
    
    asyncio.run(run_collector())

@cli.command()
@click.option('--output', '-o', default='telegram_messages_export',
              help='Output filename without extension (default: telegram_messages_export)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'excel', 'all']), default='json',
              help='Export format: json, csv, excel, or all (default: json)')
@click.option('--batch-size', '-b', default=1000, type=int,
              help='Batch size for fetching messages (default: 1000)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--summary', '-s', is_flag=True, help='Generate and display summary report')
def export(output, format, batch_size, verbose, summary):
    """Export all Telegram messages from database."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Get database URL from environment
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    logger.info(f"Connecting to database at: {db_url}")
    
    async def run_export():
        async with TelegramMessageExporter(db_url) as exporter:
            try:
                # Get total count first
                total_count = await exporter.get_total_count()
                logger.info(f"Total messages in database: {total_count}")
                
                if total_count == 0:
                    logger.warning("No messages found in database")
                    return
                
                # Fetch messages
                logger.info("Exporting all messages from database")
                messages = await exporter.get_all_messages(batch_size)
                
                if not messages:
                    logger.warning("No messages fetched")
                    return
                
                logger.info(f"Successfully fetched {len(messages)} messages")
                
                # Convert to pandas DataFrame
                df = exporter.create_dataframe(messages)
                
                # Generate summary report if requested
                if summary:
                    report = exporter.generate_summary_report(df)
                    print("\n" + report)
                
                # Export based on format
                success = False
                if format == 'json' or format == 'all':
                    json_filename = f"{output}.json"
                    success = exporter.export_to_json(df, json_filename)
                
                if format == 'csv' or format == 'all':
                    csv_filename = f"{output}.csv"
                    success = exporter.export_to_csv(df, csv_filename)
                
                if format == 'excel' or format == 'all':
                    excel_filename = f"{output}.xlsx"
                    success = exporter.export_to_excel(df, excel_filename)
                
                if success:
                    logger.info(f"Export completed successfully!")
                    if format == 'all':
                        logger.info(f"Output files: {output}.json, {output}.csv, {output}.xlsx")
                    else:
                        logger.info(f"Output file: {output}.{format}")
                    logger.info(f"Total messages exported: {len(messages)}")
                    
                    # Show DataFrame info
                    logger.info(f"DataFrame shape: {df.shape}")
                    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                else:
                    logger.error("Export failed!")
                    
            except Exception as e:
                logger.error(f"Export failed with error: {e}")
                sys.exit(1)
    
    asyncio.run(run_export())

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--dashboard', '-d', is_flag=True, help='Generate interactive HTML dashboard')
@click.option('--output', '-o', default='telegram_analysis_dashboard.html',
              help='Output HTML file path for dashboard (default: telegram_analysis_dashboard.html)')
def analyze(data_file, verbose, dashboard, output):
    """Analyze Telegram message data from a file with console reports and optional interactive HTML dashboard."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info(f"Analyzing data from: {data_file}")
    
    analyzer = TelegramDataAnalyzer(data_file)
    
    if analyzer.df is not None:
        # Generate console report
        analyzer.generate_full_report()
        
        # Generate HTML dashboard if requested
        if dashboard:
            logger.info("Generating interactive HTML dashboard...")
            dashboard_file = analyzer.generate_html_dashboard(output)
            if dashboard_file:
                logger.info(f"✅ Dashboard generated successfully!")
                logger.info(f"📁 Output file: {dashboard_file}")
                logger.info(f"🌐 Open the HTML file in your browser to view the dashboard")
            else:
                logger.warning("Dashboard generation failed. Check if plotly is installed: pip install plotly")
        else:
            logger.info("Use --dashboard flag to generate interactive HTML dashboard")
    else:
        logger.error("Failed to load data for analysis")
        sys.exit(1)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--export-csv', '-e', is_flag=True, help='Export duplicate analysis to CSV')
@click.option('--output', '-o', default='duplicate_documents.csv',
              help='Output CSV file path for duplicate analysis (default: duplicate_documents.csv)')
def analyze_docname(data_file, verbose, export_csv, output):
    """Analyze PDF and EPUB document filenames for uniqueness and duplicates."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info(f"Analyzing document names from: {data_file}")
    
    analyzer = DocumentNameAnalyzer(data_file)
    
    if analyzer.df is not None:
        # Run document name analysis
        analysis = analyzer.analyze_document_names()
        
        if not analysis:
            logger.error("No document analysis could be performed")
            sys.exit(1)
        
        # Analyze duplicates
        duplicates_analysis = analyzer.analyze_duplicates(analysis)
        
        # Generate report
        analyzer.generate_report(analysis, duplicates_analysis)
        
        # Export CSV if requested
        if export_csv and duplicates_analysis:
            csv_file = analyzer.export_duplicates_csv(duplicates_analysis, output)
            logger.info(f"✅ Duplicate analysis exported to: {csv_file}")
        elif export_csv:
            logger.warning("No duplicate data to export")
        
        logger.info("✅ Document filename analysis completed successfully!")
    else:
        logger.error("Failed to load data for analysis")
        sys.exit(1)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--batch-size', '-b', default=100, type=int, help='Batch size for importing messages (default: 100)')
@click.option('--dry-run', is_flag=True, help='Show what would be imported without actually importing')
@click.option('--skip-duplicates', is_flag=True, help='Skip messages that already exist in database')
@click.option('--validate-only', is_flag=True, help='Only validate data format without importing')
@click.option('--max-retries', '-r', default=5, type=int, help='Maximum retry attempts for failed operations (default: 5)')
@click.option('--retry-delay', '-d', default=2.0, type=float, help='Base delay between retries in seconds (default: 2.0)')
@click.option('--max-delay', '-m', default=60.0, type=float, help='Maximum delay between retries in seconds (default: 60.0)')
@click.option('--batch-delay', default=1.0, type=float, help='Delay between batches in seconds (default: 1.0)')
@click.option('--limit', '-l', default=None, type=int, help='Limit the number of records to import (default: no limit)')
@click.option('--check-quality', is_flag=True, help='Check data quality and report potential issues before import')
def import_data(data_file, verbose, batch_size, dry_run, skip_duplicates, validate_only, max_retries, retry_delay, max_delay, batch_delay, limit, check_quality):
    """Import Telegram message data from JSON or CSV file into the database via API.
    
    The import process includes automatic retry logic with exponential backoff for failed operations.
    Use the --limit option to import only a subset of records for testing purposes.
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info(f"Importing data from: {data_file}")
    
    # Get database URL from environment
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    logger.info(f"Connecting to database at: {db_url}")
    
    async def run_import():
        try:
            # Load and validate data
            if data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle structured JSON format
                if isinstance(data, dict) and 'data' in data:
                    messages = data['data']
                    logger.info(f"Found structured JSON with {len(messages)} messages")
                elif isinstance(data, dict) and 'messages' in data:
                    messages = data['messages']
                    logger.info(f"Found JSON with messages array: {len(messages)} messages")
                elif isinstance(data, list):
                    messages = data
                    logger.info(f"Found JSON array with {len(messages)} messages")
                else:
                    logger.error("Unsupported JSON format")
                    return
                    
            elif data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
                messages = df.to_dict('records')
                logger.info(f"Loaded CSV with {len(messages)} messages")
            else:
                logger.error("Unsupported file format. Use .json or .csv files")
                return
            
            if not messages:
                logger.error("No messages found in file")
                return
            
            # Apply limit if specified
            if limit is not None:
                if limit <= 0:
                    logger.error("Limit must be a positive number")
                    return
                if limit > len(messages):
                    logger.warning(f"Limit ({limit}) is greater than available messages ({len(messages)}). Using all available messages.")
                    actual_limit = len(messages)
                else:
                    actual_limit = limit
                
                original_count = len(messages)
                messages = messages[:actual_limit]
                logger.info(f"Limited import to {len(messages)} messages (from {original_count} total)")
            else:
                logger.info(f"Total messages to process: {len(messages)}")
            
            # Validate data format and quality
            if validate_only or check_quality:
                logger.info("VALIDATION MODE - No data will be imported")
                valid_count = 0
                invalid_count = 0
                quality_issues = []
                
                for i, msg in enumerate(messages):
                    if validate_message_format(msg):
                        valid_count += 1
                        
                        # Check data quality if requested
                        if check_quality:
                            issues = check_data_quality(msg)
                            if issues:
                                quality_issues.extend([f"Message {i}: {issue}" for issue in issues])
                    else:
                        invalid_count += 1
                        if verbose:
                            logger.warning(f"Invalid message at index {i}: {msg}")
                
                logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid messages")
                
                if check_quality and quality_issues:
                    logger.warning(f"Found {len(quality_issues)} data quality issues:")
                    for issue in quality_issues[:10]:  # Show first 10 issues
                        logger.warning(f"  {issue}")
                    if len(quality_issues) > 10:
                        logger.warning(f"  ... and {len(quality_issues) - 10} more issues")
                elif check_quality:
                    logger.info("No data quality issues found")
                
                if validate_only:
                    return
            
            # Initialize retry handler
            retry_handler = RetryHandler(max_retries=max_retries, base_delay=retry_delay, max_delay=max_delay)
            
            # Initialize database service
            async with TelegramDBService(db_url) as db_service:
                imported_count = 0
                skipped_count = 0
                error_count = 0
                consecutive_failures = 0
                max_consecutive_failures = 10
                
                # Check initial connection
                if not dry_run:
                    logger.info("Checking database connection...")
                    if not await db_service.check_connection():
                        logger.error("Cannot connect to database. Please check if the service is running.")
                        return
                    logger.info("Database connection successful")
                
                # Process messages in batches
                for i in range(0, len(messages), batch_size):
                    batch = messages[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(messages) + batch_size - 1) // batch_size
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)")
                    
                    batch_success_count = 0
                    batch_error_count = 0
                    
                    for msg in batch:
                        try:
                            # Validate message format
                            if not validate_message_format(msg):
                                logger.warning(f"Skipping invalid message: {msg.get('message_id', 'unknown')}")
                                error_count += 1
                                batch_error_count += 1
                                continue
                            
                            # Check for duplicates if requested
                            if skip_duplicates:
                                # This would require a check_duplicate method in TelegramDBService
                                # For now, we'll skip this feature
                                pass
                            
                            if not dry_run:
                                # Convert message to TelegramMessage format
                                telegram_msg = convert_dict_to_telegram_message(msg)
                                if telegram_msg:
                                    # Use retry handler for storing messages
                                    success = await retry_handler.execute_with_retry(
                                        db_service.store_message, telegram_msg
                                    )
                                    if success:
                                        imported_count += 1
                                        batch_success_count += 1
                                        consecutive_failures = 0  # Reset failure counter on success
                                    else:
                                        error_count += 1
                                        batch_error_count += 1
                                        consecutive_failures += 1
                                else:
                                    error_count += 1
                                    batch_error_count += 1
                                    consecutive_failures += 1
                            else:
                                # Dry run - just count
                                imported_count += 1
                                batch_success_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            error_count += 1
                            batch_error_count += 1
                            consecutive_failures += 1
                    
                    # Check if we're hitting too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}). Pausing for 30 seconds...")
                        await asyncio.sleep(30)
                        consecutive_failures = 0
                    
                    # Progress update
                    logger.info(f"Batch {batch_num} complete. Success: {batch_success_count}, Errors: {batch_error_count}")
                    if verbose:
                        logger.info(f"Progress: {min(i + batch_size, len(messages))}/{len(messages)} messages")
                    
                    # Add delay between batches to reduce load
                    if batch_delay > 0 and i + batch_size < len(messages):
                        logger.debug(f"Waiting {batch_delay}s before next batch...")
                        await asyncio.sleep(batch_delay)
                
                # Final summary
                if dry_run:
                    logger.info(f"DRY RUN COMPLETE - Would import {imported_count} messages")
                else:
                    logger.info(f"IMPORT COMPLETE - Imported {imported_count} messages")
                
                logger.info(f"Summary: {imported_count} imported, {skipped_count} skipped, {error_count} errors")
                logger.info(f"Retry statistics: {retry_handler.retry_count} total retries performed")
                
        except Exception as e:
            logger.error(f"Import failed with error: {e}")
            sys.exit(1)
    
    asyncio.run(run_import())

def validate_message_format(msg):
    """Validate that a message has the required fields for import."""
    required_fields = ['message_id', 'channel_username', 'text']
    
    for field in required_fields:
        if field not in msg or msg[field] is None:
            return False
    
    return True

def check_data_quality(msg):
    """Check data quality and identify potential JSON serialization issues."""
    issues = []
    
    for key, value in msg.items():
        if value is not None:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(f"Field '{key}' contains invalid float value: {value}")
            elif isinstance(value, (datetime, timedelta)):
                # These will be converted during processing
                pass
            elif not isinstance(value, (str, int, bool, list, dict)):
                issues.append(f"Field '{key}' has non-serializable type: {type(value).__name__}")
    
    return issues

def convert_dict_to_telegram_message(msg_dict):
    """Convert a dictionary to TelegramMessage object."""
    try:
        # Handle date conversion
        date_str = msg_dict.get('date')
        if date_str:
            try:
                if isinstance(date_str, str):
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    date_obj = date_str
            except:
                date_obj = datetime.now()
        else:
            date_obj = datetime.now()
        
        # Create TelegramMessage object
        telegram_msg = TelegramMessage(
            message_id=msg_dict['message_id'],
            channel_username=msg_dict['channel_username'],
            date=date_obj,
            text=msg_dict['text'],
            media_type=msg_dict.get('media_type'),
            file_name=msg_dict.get('file_name'),
            file_size=msg_dict.get('file_size'),
            mime_type=msg_dict.get('mime_type'),
            caption=msg_dict.get('caption'),
            views=msg_dict.get('views'),
            forwards=msg_dict.get('forwards'),
            replies=msg_dict.get('replies'),
            is_forwarded=msg_dict.get('is_forwarded', False),
            forwarded_from=msg_dict.get('forwarded_from'),
            creator_username=msg_dict.get('creator_username'),
            creator_first_name=msg_dict.get('creator_first_name'),
            creator_last_name=msg_dict.get('creator_last_name'),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return telegram_msg
        
    except Exception as e:
        logger.error(f"Error converting message: {e}")
        return None

if __name__ == '__main__':
    cli()

