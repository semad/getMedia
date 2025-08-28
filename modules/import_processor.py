"""
Import processor for Telegram messages.
"""

import json
import logging
import math
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import TelegramMessage, ImportStats
from .database_service import TelegramDBService
from .retry_handler import RetryHandler

logger = logging.getLogger(__name__)


def validate_message_format(msg: Dict[str, Any]) -> bool:
    """Validate that a message has the required fields for import."""
    required_fields = ['message_id', 'channel_username', 'text']
    
    for field in required_fields:
        if field not in msg or msg[field] is None:
            return False
    
    return True


def check_data_quality(msg: Dict[str, Any]) -> List[str]:
    """Check data quality and identify potential JSON serialization issues."""
    issues = []
    
    for key, value in msg.items():
        if value is not None:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(f"Field '{key}' contains invalid float value: {value}")
            elif isinstance(value, (datetime, type)):
                issues.append(f"Field '{key}' contains non-serializable type: {type(value).__name__}")
    
    return issues


def load_messages_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load messages from JSON or CSV file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        else:
            raise ValueError("Invalid JSON structure. Expected list of messages or dict with 'messages' key.")
            
        logger.info(f"Found JSON with messages array: {len(messages)} messages")
        return messages
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


async def process_import_batch(
    batch: List[Dict[str, Any]], 
    db_service: TelegramDBService, 
    retry_handler: RetryHandler,
    skip_duplicates: bool = False
) -> tuple[int, int, int]:
    """Process a batch of messages for import."""
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for msg in batch:
        try:
            # Validate message format
            if not validate_message_format(msg):
                logger.warning(f"Skipping invalid message: {msg}")
                error_count += 1
                continue
            
            # Convert dict to TelegramMessage object
            message = TelegramMessage(
                message_id=msg['message_id'],
                channel_username=msg['channel_username'],
                date=datetime.fromisoformat(msg['date']) if isinstance(msg['date'], str) else msg['date'],
                text=msg['text'],
                media_type=msg.get('media_type'),
                file_name=msg.get('file_name'),
                file_size=msg.get('file_size'),
                mime_type=msg.get('mime_type'),
                duration=msg.get('duration'),
                width=msg.get('width'),
                height=msg.get('height'),
                caption=msg.get('caption'),
                views=msg.get('views'),
                forwards=msg.get('forwards'),
                replies=msg.get('replies'),
                edit_date=datetime.fromisoformat(msg['edit_date']) if msg.get('edit_date') else None,
                is_forwarded=msg.get('is_forwarded', False),
                forwarded_from=msg.get('forwarded_from'),
                forwarded_message_id=msg.get('forwarded_message_id'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store message with retry logic
            success = await retry_handler.execute_with_retry(
                db_service.store_message, message
            )
            
            if success:
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_count += 1
    
    return success_count, error_count, skipped_count


async def run_import(
    data_file: str,
    db_url: str,
    batch_size: int = 100,
    dry_run: bool = False,
    skip_duplicates: bool = False,
    validate_only: bool = False,
    check_quality: bool = False,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    max_delay: float = 60.0,
    batch_delay: float = 1.0,
    limit: Optional[int] = None
) -> ImportStats:
    """Run the import process."""
    stats = ImportStats()
    stats.start_time = datetime.now()
    
    try:
        # Load messages from file
        messages = load_messages_from_file(data_file)
        stats.total_messages = len(messages)
        
        if not messages:
            logger.error("No messages found in file")
            return stats
        
        # Apply limit if specified
        if limit is not None:
            if limit <= 0:
                logger.error("Limit must be a positive number")
                return stats
            if limit > len(messages):
                logger.warning(f"Limit ({limit}) is greater than available messages ({len(messages)}). Using all available messages.")
                limit = len(messages)
            
            original_count = len(messages)
            messages = messages[:limit]
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
                    if check_quality:
                        issues = check_data_quality(msg)
                        quality_issues.extend([f"Message {i}: {issue}" for issue in issues])
                else:
                    invalid_count += 1
            
            logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid messages")
            
            if quality_issues:
                logger.warning(f"Found {len(quality_issues)} data quality issues:")
                for issue in quality_issues[:10]:  # Show first 10 issues
                    logger.warning(f"  {issue}")
                if len(quality_issues) > 10:
                    logger.warning(f"  ... and {len(quality_issues) - 10} more issues")
            
            return stats
        
        # Initialize database service
        async with TelegramDBService(db_url) as db_service:
            # Check connection
            logger.info("Checking database connection...")
            if not await db_service.check_connection():
                logger.error("Cannot connect to database. Please check if the service is running.")
                return stats
            
            logger.info("Database connection successful")
            
            # Initialize retry handler
            retry_handler = RetryHandler(max_retries, retry_delay, max_delay)
            
            # Process messages in batches
            total_batches = (len(messages) + batch_size - 1) // batch_size
            
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)")
                
                if not dry_run:
                    success, errors, skipped = await process_import_batch(
                        batch, db_service, retry_handler, skip_duplicates
                    )
                    
                    stats.imported_count += success
                    stats.error_count += errors
                    stats.skipped_count += skipped
                    
                    logger.info(f"Batch {batch_num} complete. Success: {success}, Errors: {errors}")
                    
                    # Progress update
                    progress = min(i + batch_size, len(messages))
                    logger.info(f"Progress: {progress}/{len(messages)} messages")
                    
                    # Add batch delay if not the last batch
                    if batch_num < total_batches and batch_delay > 0:
                        logger.debug(f"Waiting {batch_delay}s before next batch...")
                        await asyncio.sleep(batch_delay)
                else:
                    logger.info(f"DRY RUN - Would process batch {batch_num} ({len(batch)} messages)")
                    stats.imported_count += len(batch)
            
            # Update retry statistics
            stats.retry_count = retry_handler.total_retries
    
    except Exception as e:
        logger.error(f"Import failed: {e}")
        stats.error_count += 1
    
    finally:
        stats.end_time = datetime.now()
    
    return stats
