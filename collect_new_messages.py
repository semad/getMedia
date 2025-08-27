#!/usr/bin/env python3
"""
Script to check database and collect new Telegram messages
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Set
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseChecker:
    """Check database for existing messages."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.session: aiohttp.ClientSession = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_existing_message_ids(self, channel_username: str) -> Set[int]:
        """Get existing message IDs from database."""
        try:
            url = f"{self.db_url}/api/v1/telegram/messages"
            params = {"channel_username": channel_username, "limit": 1000}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list):
                        return {msg.get('message_id') for msg in data if msg.get('message_id')}
                    else:
                        logger.warning(f"Unexpected response format: {data}")
                        return set()
                else:
                    logger.warning(f"Failed to get messages: {response.status}")
                    return set()
                    
        except Exception as e:
            logger.error(f"Error getting existing messages: {e}")
            return set()
    
    async def get_database_stats(self) -> dict:
        """Get database statistics."""
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

async def main():
    """Main function to check database and collect new messages."""
    
    # Load configuration
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    
    logger.info("Checking database for existing messages...")
    
    async with DatabaseChecker(db_url) as checker:
        # Get database stats
        stats = await checker.get_database_stats()
        logger.info(f"Database stats: {stats}")
        
        # Get existing message IDs for the channel
        channel_username = "@SherwinVakiliLibrary"
        existing_ids = await checker.get_existing_message_ids(channel_username)
        
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing messages in database")
            logger.info(f"Existing message IDs: {sorted(existing_ids)}")
        else:
            logger.info("No existing messages found in database")
        
        # Now run the collector to get new messages
        logger.info("Starting collection of new messages...")
        
        # Run the enhanced collector script
        import subprocess
        import sys
        
        # Build command with appropriate options
        cmd = [
            sys.executable, "0_getMediaFromTel_enhanced.py",
            "--max-messages", "100",  # Collect more messages to find new ones
            "--no-download-media",    # Don't download media yet
            "--save-json",           # Save to JSON for backup
            "--json-filename", "new_messages.json"
        ]
        
        if existing_ids:
            logger.info("Note: The collector will automatically handle duplicates")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Message collection completed successfully")
                logger.info("STDOUT:", result.stdout)
            else:
                logger.error("Message collection failed")
                logger.error("STDERR:", result.stderr)
        except Exception as e:
            logger.error(f"Error running collector: {e}")

if __name__ == '__main__':
    asyncio.run(main())
