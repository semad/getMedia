#!/usr/bin/env python3
"""
Telegram Messages Database Export Script

This script exports all messages from the telegram_messages table
into a JSON file for backup, analysis, or migration purposes.

Usage:
    python export_telegram_messages.py [--output filename.json] [--format pretty|compact]
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
import aiohttp
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramMessageExporter:
    """Export Telegram messages from database to JSON."""
    
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
    
    async def get_total_count(self) -> int:
        """Get total number of messages in database."""
        try:
            url = f"{self.db_url}/api/v1/telegram/stats"
            async with self.session.get(url) as response:
                if response.status == 200:
                    stats = await response.json()
                    return stats.get('total_messages', 0)
                else:
                    logger.warning(f"Failed to get stats: {response.status}")
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
                        
                        # Handle paginated response
                        if isinstance(data, dict) and 'data' in data:
                            messages = data.get('data', [])
                            total_count = data.get('total_count', 0)
                        else:
                            # Fallback for non-paginated response
                            messages = data if isinstance(data, list) else []
                            total_count = len(messages)
                        
                        if not messages:
                            logger.info("No more messages to fetch")
                            break
                        
                        all_messages.extend(messages)
                        logger.info(f"Fetched {len(messages)} messages (total so far: {len(all_messages)}/{total_count})")
                        
                        # Check if we've fetched all messages
                        if len(all_messages) >= total_count:
                            logger.info("All messages fetched")
                            break
                        
                        page += 1
                    else:
                        logger.error(f"Failed to fetch batch {page}: {response.status}")
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
        
        return all_messages
    
    async def get_messages_by_channel(self, channel_username: str, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Get all messages from a specific channel."""
        all_messages = []
        page = 1
        
        try:
            while True:
                logger.info(f"Fetching batch {page} for channel {channel_username} (batch size: {batch_size})...")
                
                url = f"{self.db_url}/api/v1/telegram/messages"
                params = {
                    "channel_username": channel_username,
                    "page": page,
                    "items_per_page": batch_size
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle paginated response
                        if isinstance(data, dict) and 'data' in data:
                            messages = data.get('data', [])
                            total_count = data.get('total_count', 0)
                        else:
                            # Fallback for non-paginated response
                            messages = data if isinstance(data, list) else []
                            total_count = len(messages)
                        
                        if not messages:
                            logger.info(f"No more messages for channel {channel_username}")
                            break
                        
                        all_messages.extend(messages)
                        logger.info(f"Fetched {len(messages)} messages for {channel_username} (total so far: {len(all_messages)}/{total_count})")
                        
                        # Check if we've fetched all messages
                        if len(all_messages) >= total_count:
                            logger.info(f"All messages for channel {channel_username} fetched")
                            break
                        
                        page += 1
                    else:
                        logger.error(f"Failed to fetch batch {page} for channel {channel_username}: {response.status}")
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching messages for channel {channel_username}: {e}")
        
        return all_messages
    
    def format_messages(self, messages: List[Dict[str, Any]], format_type: str = "pretty") -> str:
        """Format messages for JSON output."""
        export_data = {
            "export_info": {
                "exported_at": datetime.now(UTC).isoformat(),
                "total_messages": len(messages),
                "format": format_type,
                "source": "telegram_messages_table"
            },
            "messages": messages
        }
        
        if format_type == "pretty":
            return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        else:
            return json.dumps(export_data, ensure_ascii=False, default=str)
    
    def save_to_file(self, data: str, filename: str) -> bool:
        """Save data to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(data)
            logger.info(f"Data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving to file {filename}: {e}")
            return False


@click.command(help='Export Telegram messages from database to JSON')
@click.option('--output', '-o', default='telegram_messages_export.json',
              help='Output JSON filename (default: telegram_messages_export.json)')
@click.option('--format', '-f', type=click.Choice(['pretty', 'compact']), default='pretty',
              help='JSON format: pretty (indented) or compact (default: pretty)')
@click.option('--channel', '-c', help='Export only messages from specific channel username')
@click.option('--batch-size', '-b', default=1000, type=int,
              help='Batch size for fetching messages (default: 1000)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(output: str, format: str, channel: str, batch_size: int, verbose: bool):
    """Export all Telegram messages from database to JSON file."""
    
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
                if channel:
                    logger.info(f"Exporting messages from channel: {channel}")
                    messages = await exporter.get_messages_by_channel(channel, batch_size)
                else:
                    logger.info("Exporting all messages from database")
                    messages = await exporter.get_all_messages(batch_size)
                
                if not messages:
                    logger.warning("No messages fetched")
                    return
                
                logger.info(f"Successfully fetched {len(messages)} messages")
                
                # Format and save
                formatted_data = exporter.format_messages(messages, format)
                if exporter.save_to_file(formatted_data, output):
                    logger.info(f"Export completed successfully!")
                    logger.info(f"Output file: {output}")
                    logger.info(f"Total messages exported: {len(messages)}")
                    
                    # Show file size
                    file_size = os.path.getsize(output)
                    logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                else:
                    logger.error("Export failed!")
                    
            except Exception as e:
                logger.error(f"Export failed with error: {e}")
                sys.exit(1)
    
    # Run the export
    asyncio.run(run_export())


if __name__ == '__main__':
    main()
