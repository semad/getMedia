#!/usr/bin/env python3
"""
Diff Utility for Telegram Messages

This module provides functionality to compare a JSON export file with the database
and generate a diff JSON containing only the messages that are missing from the database.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import aiohttp
import math

logger = logging.getLogger(__name__)


class DatabaseDiffChecker:
    """Utility class to check differences between JSON export and database."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_database_messages(self) -> Dict[str, Set[int]]:
        """Get all message IDs from database grouped by channel."""
        try:
            # Get stats first to check connection
            stats_url = f"{self.db_url}/api/v1/telegram/stats"
            async with self.session.get(stats_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to get stats: {response.status}")
                    return {}
                
                stats = await response.json()
                logger.info(f"Database stats: {stats}")
            
            # Get all messages from database
            messages_url = f"{self.db_url}/api/v1/telegram/messages"
            all_messages = []
            page = 1
            items_per_page = 1000
            
            while True:
                params = {
                    'page': page,
                    'items_per_page': items_per_page
                }
                
                async with self.session.get(messages_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get messages page {page}: {response.status}")
                        break
                    
                    data = await response.json()
                    messages = data.get('data', [])  # API returns 'data' not 'items'
                    
                    if not messages:
                        break
                    
                    all_messages.extend(messages)
                    logger.info(f"Retrieved page {page}: {len(messages)} messages")
                    
                    # Check if there are more pages
                    has_more = data.get('has_more', False)
                    if not has_more:
                        break
                    
                    page += 1
            
            # Group by channel and extract message IDs
            db_messages: Dict[str, Set[int]] = {}
            for msg in all_messages:
                channel = msg.get('channel_username')
                message_id = msg.get('message_id')
                if channel and message_id:
                    if channel not in db_messages:
                        db_messages[channel] = set()
                    db_messages[channel].add(message_id)
            
            logger.info(f"Retrieved {len(all_messages)} messages from database")
            for channel, ids in db_messages.items():
                logger.info(f"Channel {channel}: {len(ids)} messages")
            
            return db_messages
            
        except Exception as e:
            logger.error(f"Error getting database messages: {e}")
            return {}
    
    def load_json_export(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load and parse the JSON export file, removing duplicates."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'messages' not in data:
                raise ValueError("JSON file does not contain 'messages' key")
            
            # Group messages by channel and remove duplicates
            grouped_messages: Dict[str, List[Dict[str, Any]]] = {}
            seen_messages: Dict[str, Set[int]] = {}  # Track seen message IDs per channel
            
            for msg in data['messages']:
                channel = msg.get('channel_username')
                message_id = msg.get('message_id')
                
                if channel and message_id is not None:
                    if channel not in grouped_messages:
                        grouped_messages[channel] = []
                        seen_messages[channel] = set()
                    
                    # Only add if we haven't seen this message ID in this channel
                    if message_id not in seen_messages[channel]:
                        seen_messages[channel].add(message_id)
                        grouped_messages[channel].append(msg)
            
            # Log deduplication results
            total_original = len(data['messages'])
            total_deduplicated = sum(len(msgs) for msgs in grouped_messages.values())
            duplicates_removed = total_original - total_deduplicated
            
            logger.info(f"Loaded {total_original} messages from JSON export")
            logger.info(f"After deduplication: {total_deduplicated} messages")
            logger.info(f"Duplicates removed: {duplicates_removed} messages ({duplicates_removed/total_original*100:.1f}%)")
            
            for channel, msgs in grouped_messages.items():
                logger.info(f"Channel {channel}: {len(msgs)} messages (deduplicated)")
            
            return grouped_messages
            
        except Exception as e:
            logger.error(f"Error loading JSON export: {e}")
            return {}
    
    def clean_message_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Clean message data for import, removing problematic fields and values."""
        # Fields to keep (only essential ones for import)
        allowed_fields = {
            'message_id', 'channel_username', 'date', 'text', 'media_type',
            'file_name', 'file_size', 'mime_type', 'caption'
        }
        
        cleaned_msg = {}
        for key, value in msg.items():
            if key in allowed_fields and value is not None:
                # Handle NaN and inf values
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    continue  # Skip this field
                elif isinstance(value, str) and value in ['NaN', 'NaT', 'nan', 'nat']:
                    continue  # Skip this field
                else:
                    # Convert float values to int for integer fields
                    if key == 'file_size' and isinstance(value, float):
                        cleaned_msg[key] = int(value)
                    else:
                        cleaned_msg[key] = value
        
        return cleaned_msg
    
    def find_missing_messages(self, 
                             json_messages: Dict[str, List[Dict[str, Any]]], 
                             db_messages: Dict[str, Set[int]]) -> List[Dict[str, Any]]:
        """Find messages that exist in JSON but not in database."""
        missing_messages = []
        total_checked = 0
        total_found_in_db = 0
        
        for channel, messages in json_messages.items():
            db_ids = db_messages.get(channel, set())
            channel_missing = 0
            channel_found = 0
            
            logger.info(f"Checking channel {channel}: {len(messages)} in JSON, {len(db_ids)} in DB")
            
            for msg in messages:
                message_id = msg.get('message_id')
                total_checked += 1
                
                if message_id and message_id not in db_ids:
                    # Clean the message data
                    cleaned_msg = self.clean_message_data(msg)
                    
                    # Only include if it has required fields
                    if ('message_id' in cleaned_msg and 
                        'channel_username' in cleaned_msg and 
                        'text' in cleaned_msg):
                        missing_messages.append(cleaned_msg)
                        channel_missing += 1
                else:
                    channel_found += 1
                    total_found_in_db += 1
            
            logger.info(f"  Channel {channel}: {channel_missing} missing, {channel_found} found in DB")
        
        logger.info(f"=== COMPARISON SUMMARY ===")
        logger.info(f"Total messages checked: {total_checked}")
        logger.info(f"Total found in database: {total_found_in_db}")
        logger.info(f"Total missing: {len(missing_messages)}")
        logger.info(f"Missing percentage: {len(missing_messages)/total_checked*100:.1f}%")
        
        return missing_messages
    
    async def generate_diff(self, json_file_path: str, output_file_path: str) -> Dict[str, Any]:
        """Generate a diff JSON file with missing messages."""
        logger.info("Starting diff generation...")
        
        # Load JSON export
        json_messages = self.load_json_export(json_file_path)
        if not json_messages:
            logger.error("Failed to load JSON export")
            return {}
        
        # Get database messages
        db_messages = await self.get_database_messages()
        if not db_messages:
            logger.error("Failed to get database messages")
            return {}
        
        # Find missing messages
        missing_messages = self.find_missing_messages(json_messages, db_messages)
        logger.info(f"Found {len(missing_messages)} missing messages")
        
        # Generate diff summary
        diff_summary = {
            'generated_at': datetime.now().isoformat(),
            'source_file': json_file_path,
            'database_url': self.db_url,
            'summary': {
                'total_in_json': sum(len(msgs) for msgs in json_messages.values()),
                'total_in_database': sum(len(ids) for ids in db_messages.values()),
                'missing_messages': len(missing_messages),
                'channels_in_json': list(json_messages.keys()),
                'channels_in_database': list(db_messages.keys())
            },
            'missing_messages': missing_messages
        }
        
        # Save diff file
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(diff_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Diff file saved to: {output_file_path}")
        except Exception as e:
            logger.error(f"Error saving diff file: {e}")
            return {}
        
        return diff_summary
