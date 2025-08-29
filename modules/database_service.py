"""
Database service for storing Telegram messages.
"""

import aiohttp
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List
from dataclasses import asdict

from .models import TelegramMessage

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
    
    async def store_message(self, message: Union[TelegramMessage, Dict[str, Any]]) -> bool:
        """Store a Telegram message in the database."""
        try:
            logger.debug(f"Starting to store message...")
            
            # Handle both TelegramMessage objects and dictionaries
            if isinstance(message, TelegramMessage):
                message_data = asdict(message)
                message_id = message.message_id
                logger.debug(f"Processing TelegramMessage object with ID: {message_id}")
            else:
                message_data = message.copy()
                message_id = message_data.get('message_id', 'unknown')
                logger.debug(f"Processing dictionary message with ID: {message_id}")
            
            logger.debug(f"Original message data keys: {list(message_data.keys())}")
            
            # Remove fields that are not in the API schema
            fields_to_remove = ['created_at', 'updated_at', 'id', 'uuid', 'deleted_at', 'is_deleted']
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
        """Store multiple Telegram messages using the bulk import endpoint."""
        try:
            logger.debug(f"Starting bulk import of {len(messages)} messages...")
            logger.debug(f"Skip duplicates setting: {skip_duplicates}")
            
            # Clean and prepare messages for bulk import
            cleaned_messages = []
            for i, msg in enumerate(messages):
                logger.debug(f"Processing message {i+1}/{len(messages)}: {msg.get('message_id', 'unknown')}")
                # Only include the most essential fields to avoid validation issues
                allowed_fields = {
                    'message_id', 'channel_username', 'date', 'text', 'media_type',
                    'file_name', 'file_size', 'mime_type', 'caption'
                }
                
                cleaned_msg = {}
                for key, value in msg.items():
                    if key in allowed_fields and value is not None:
                        # Clean NaN and inf values
                        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                            continue  # Skip this field
                        elif isinstance(value, str) and value in ['NaN', 'NaT', 'nan', 'nat']:
                            continue  # Skip this field
                        else:
                            # Convert float values to int for integer fields
                            if key in ['file_size', 'views', 'forwards', 'replies'] and isinstance(value, float):
                                cleaned_msg[key] = int(value)
                            else:
                                cleaned_msg[key] = value
                
                # Ensure required fields are present
                if 'message_id' in cleaned_msg and 'channel_username' in cleaned_msg and 'text' in cleaned_msg:
                    cleaned_messages.append(cleaned_msg)
                else:
                    logger.warning(f"Skipping message with missing required fields: {cleaned_msg}")
            
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
            url = f"{self.db_url}/api/v1/telegram/stats"
            if self.session:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return None
