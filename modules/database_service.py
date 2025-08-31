"""
Database service for storing Telegram messages.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_ENDPOINTS
import aiohttp
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List




logger = logging.getLogger(__name__)


class TelegramDBService:
    """Service for storing Telegram messages in the database via API."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
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
    
    def clean_message_data(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean message data to ensure JSON serialization compatibility."""
        cleaned_data: Dict[str, Any] = {}
        
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
                # Convert other types to string or None
                try:
                    cleaned_data[key] = str(value)
                except:
                    cleaned_data[key] = None
        
        return cleaned_data
    
    async def store_message(self, message: Dict[str, Any]) -> bool:
        """Store a Telegram message in the database."""
        try:
            logger.debug(f"Starting to store message...")
            
            message_data = message.copy()
            message_id = message_data.get('message_id', 'unknown')
            logger.debug(f"Processing message with ID: {message_id}")
            
            logger.debug(f"Original message data keys: {list(message_data.keys())}")
            
            # Remove fields that are not in the API schema
            fields_to_remove = ['created_at', 'updated_at', 'id', 'uuid', 'deleted_at', 'is_deleted', 'duration', 'width', 'height', 'edit_date', 'forwarded_message_id']
            for field in fields_to_remove:
                if field in message_data:
                    logger.debug(f"Removing field: {field}")
                    message_data.pop(field, None)
            
            logger.debug(f"Message data after field removal: {list(message_data.keys())}")
            
            # Clean the data for JSON serialization
            logger.debug("Cleaning message data for JSON serialization...")
            message_data = self.clean_message_data(message_data)
            logger.debug(f"Cleaned message data: {message_data}")
            
            # Send to microservice
            url = f"{self.db_url}/api/v1/telegram/messages"
            logger.debug(f"Sending message to API endpoint: {url}")
            if self.session:
                async with self.session.post(url, json=message_data) as response:
                    response_text = await response.text()
                    logger.debug(f"API Response for message {message_id}: Status {response.status}, Body: {response_text[:200]}")
                    
                    if response.status in [200, 201]:  # Accept both 200 (OK) and 201 (Created)
                        logger.debug(f"Stored message {message_id} in database")
                        return True
                    else:
                        logger.warning(f"Failed to store message {message_id}: {response.status} - {response_text}")
                        return False
            else:
                logger.error("No active session")
                return False
                
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            return False

    async def store_messages_bulk(self, messages: List[Dict[str, Any]], skip_duplicates: bool = True) -> Dict[str, Any]:
        """Store multiple Telegram messages in bulk."""
        try:
            logger.debug(f"Starting bulk import of {len(messages)} messages...")
            logger.debug(f"Skip duplicates setting: {skip_duplicates}")
            
            cleaned_messages = []
            
            for message in messages:
                try:
                    # Create a copy to avoid modifying the original
                    msg_copy = message.copy()
                    
                    # Remove fields that are not in the API schema
                    fields_to_remove = ['created_at', 'updated_at', 'id', 'uuid', 'deleted_at', 'is_deleted', 'duration', 'width', 'height', 'edit_date', 'forwarded_message_id']
                    for field in fields_to_remove:
                        msg_copy.pop(field, None)
                    
                    # Data is already cleaned by the import processor, just ensure required fields
                    if 'message_id' in msg_copy and 'channel_username' in msg_copy and 'text' in msg_copy:
                        cleaned_messages.append(msg_copy)
                    else:
                        logger.warning(f"Skipping message with missing required fields: {msg_copy}")
                        
                except Exception as e:
                    logger.error(f"Error processing message in bulk import: {e}")
                    continue
            
            if not cleaned_messages:
                logger.warning("No valid messages after cleaning")
                return {'error': 'No valid messages after cleaning'}
            
            # Send to bulk import endpoint
            url = f"{self.db_url}/api/v1/telegram/messages/bulk"
            params = {
                'skip_duplicates': skip_duplicates,
                'batch_size': 100
            }
            
            # Debug: Log all messages to see what we're sending
            if cleaned_messages:
                logger.debug(f"Sending {len(cleaned_messages)} messages:")
                for i, msg in enumerate(cleaned_messages):
                    # Check for any boolean values in the message
                    boolean_fields = []
                    for key, value in msg.items():
                        if isinstance(value, bool):
                            boolean_fields.append(f"{key}: {value} ({type(value).__name__})")
                    
                    if boolean_fields:
                        logger.warning(f"Message {i} contains boolean values: {boolean_fields}")
                    
                    logger.debug(f"Message {i}: {msg}")
            
            if self.session:
                async with self.session.post(url, json=cleaned_messages, params=params) as response:
                    response_text = await response.text()
                    logger.debug(f"Bulk import response: Status {response.status}, Body: {response_text[:500]}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Bulk import successful: {result.get('summary', {})}")
                        return result
                    else:
                        logger.warning(f"Bulk import failed: {response.status} - {response_text}")
                        return {'error': f'HTTP {response.status}: {response_text}'}
            else:
                logger.error("No active session")
                return {'error': 'No active session'}
                
        except Exception as e:
            logger.error(f"Error in bulk import: {e}")
            return {'error': str(e)}
    
    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get database statistics."""
        try:
            # Try enhanced stats first, fall back to basic stats if not available
            url = f"{self.db_url}/api/v1/telegram/stats/enhanced"
            if self.session:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
            
            # Fall back to basic stats
            url = f"{self.db_url}/api/v1/telegram/stats"
            if self.session:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        basic_stats = await response.json()
                        # Convert basic stats to enhanced format for compatibility
                        return {
                            "total_messages": basic_stats.get("total_messages", 0),
                            "total_channels": basic_stats.get("channels_count", 0),
                            "total_storage_gb": 0,  # Not available in basic stats
                            "media_breakdown": {},  # Not available in basic stats
                            "channel_breakdown": {},  # Not available in basic stats
                            "date_range": {"start": "N/A", "end": "N/A"},  # Not available in basic stats
                            "media_messages": basic_stats.get("media_messages", 0),
                            "text_only_messages": basic_stats.get("text_only_messages", 0)
                        }
            return None
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return None
    
    async def get_messages_for_analysis(self, limit: Optional[int] = 10000) -> Optional[List[Dict[str, Any]]]:
        """Get messages from database for comprehensive analysis."""
        try:
            # First get basic stats to know total count
            stats = await self.get_stats()
            if not stats:
                return None
            
            # If no limit specified, use the new bulk export endpoint
            if limit is None:
                total_messages = stats.get('total_messages', 0)
                if total_messages > 0:
                    logger.info(f"📊 Fetching ALL {total_messages:,} messages for comprehensive analysis...")
                    
                    # Use the bulk export endpoint for all messages
                    url = f"{self.db_url}/api/v1/telegram/messages/export/all"
                    params = {
                        'fields': 'message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at'
                    }
                    
                    if self.session:
                        async with self.session.get(url, params=params) as response:
                            if response.status == 200:
                                result = await response.json()
                                logger.info(f"✅ Successfully downloaded {len(result):,} messages")
                                return result
                            else:
                                logger.warning(f"Bulk export failed with status {response.status}, falling back to paginated approach")
                                # Fall back to paginated approach
                                limit = total_messages
                else:
                    logger.warning("No messages found in database")
                    return None
            
            # Fall back to paginated approach for specific limits
            if limit:
                logger.info(f"📊 Fetching {limit:,} messages using paginated approach...")
                url = f"{self.db_url}/api/v1/telegram/messages"
                params = {
                    'limit': min(limit, 1000),  # Respect the API limit
                    'fields': 'message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at'
                }
                
                if self.session:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'messages' in result:
                                return result['messages']
                            elif 'data' in result:
                                return result['data']
                            else:
                                return result
            
            return None
        except Exception as e:
            logger.error(f"Error getting messages for analysis: {e}")
            return None
