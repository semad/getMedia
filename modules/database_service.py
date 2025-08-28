"""
Database service for storing Telegram messages.
"""

import aiohttp
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
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
    
    async def store_message(self, message: TelegramMessage) -> bool:
        """Store a Telegram message in the database."""
        try:
            message_data = asdict(message)
            
            # Remove fields that are not in the API schema
            fields_to_remove = ['created_at', 'updated_at']
            for field in fields_to_remove:
                message_data.pop(field, None)
            
            # Clean the data for JSON serialization
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
