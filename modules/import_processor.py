"""
Import processor for Telegram messages using pandas for data handling.
"""

import asyncio
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .database_service import TelegramDBService
from .retry_handler import RetryHandler
from config import COLLECTIONS_DIR, COMBINED_COLLECTION_GLOB

logger = logging.getLogger(__name__)


def validate_message_format(msg: dict[str, Any]) -> bool:
    """Validate that a message has the required fields for import."""
    required_fields = ["message_id", "channel_username", "text"]

    for field in required_fields:
        if field not in msg or msg[field] is None:
            return False

    return True


def check_data_quality(msg: dict[str, Any]) -> list[str]:
    """Check data quality and identify potential JSON serialization issues."""
    issues = []

    for key, value in msg.items():
        if value is not None:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(
                        f"Field '{key}' contains invalid float value: {value}"
                    )
            elif isinstance(value, datetime | type):
                issues.append(
                    f"Field '{key}' contains non-serializable type: {type(value).__name__}"
                )

    return issues


def load_messages_from_file(file_path: str) -> list[dict[str, Any]]:
    """Load messages from JSON file using pandas for better data handling."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON files are supported. Found: {path.suffix}")

    try:
        # Use pandas to read JSON file with multiple strategies
        messages = []

        # Strategy 1: Try direct pandas read
        try:
            df = pd.read_json(file_path)
            logger.info(f"Successfully read JSON with pandas: {df.shape}")

            # Handle different JSON structures
            if "messages" in df.columns:
                # Structured format with metadata (our collector format)
                messages_data = (
                    df["messages"].iloc[0] if len(df) == 1 else df["messages"].tolist()
                )
                logger.info("Found structured JSON with messages in 'messages' column")
                if isinstance(messages_data, list):
                    messages = messages_data
                else:
                    messages = [messages_data]
            elif len(df) == 1 and "messages" in df.iloc[0]:
                # Single row with messages key
                messages = df.iloc[0]["messages"]
                logger.info("Found messages in single row structure")
            elif len(df) == 1 and isinstance(df.iloc[0], dict):
                # Handle combined file format: list with one dict containing metadata and messages
                row_data = df.iloc[0]
                if "messages" in row_data:
                    messages = row_data["messages"]
                    logger.info("Found combined file format with messages")
                else:
                    # Assume DataFrame contains messages directly
                    messages = df.to_dict("records")
                    logger.info(
                        f"Found direct message format with {len(messages)} messages"
                    )
            else:
                # Assume DataFrame contains messages directly
                messages = df.to_dict("records")
                logger.info(
                    f"Found direct message format with {len(messages)} messages"
                )

        except Exception as e:
            logger.warning(f"Direct pandas read failed: {e}")

            # Strategy 2: Try with lines=True for JSONL format
            try:
                df = pd.read_json(file_path, lines=True)
                messages = df.to_dict("records")
                logger.info(
                    f"Successfully read JSONL format with {len(messages)} messages"
                )
            except Exception as e2:
                logger.warning(f"JSONL read failed: {e2}")

                # Strategy 3: Try reading as string first, then parse
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Try to parse as JSON string using StringIO to avoid deprecation warning
                    from io import StringIO

                    json_content = (
                        f"[{content}]" if not content.startswith("[") else content
                    )
                    df = pd.read_json(StringIO(json_content))

                    if "messages" in df.columns:
                        messages_data = (
                            df["messages"].iloc[0]
                            if len(df) == 1
                            else df["messages"].tolist()
                        )
                        messages = (
                            messages_data
                            if isinstance(messages_data, list)
                            else [messages_data]
                        )
                    elif len(df) == 1 and "messages" in df.iloc[0]:
                        messages = df.iloc[0]["messages"]
                    else:
                        messages = df.to_dict("records")

                    logger.info(
                        f"Successfully read JSON using StringIO with {len(messages)} messages"
                    )
                except Exception as e3:
                    logger.error(f"All pandas strategies failed: {e3}")
                    raise ValueError(
                        f"Could not read JSON file with any pandas method: {e3}"
                    ) from e3

        # Validate that we have messages
        if not messages:
            raise ValueError("No messages found in file")

        # Ensure messages is a list
        if not isinstance(messages, list):
            messages = [messages]

        logger.info(f"Successfully loaded {len(messages)} messages from {file_path}")
        return messages

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def clean_message_data(msg: dict[str, Any]) -> dict[str, Any]:
    """Clean and prepare message data for API import using pandas."""
    logger.debug(
        f"Starting to clean message data for message ID: {msg.get('message_id', 'unknown')}"
    )

    # Start with required fields
    cleaned_msg = {
        "message_id": msg["message_id"],
        "channel_username": msg["channel_username"],
        "date": msg.get("date"),
        "text": msg.get("text", ""),
        "media_type": msg.get("media_type"),
        "file_name": msg.get("file_name"),
        "file_size": msg.get("file_size"),
        "mime_type": msg.get("mime_type"),
        "caption": msg.get("caption"),
    }

    # Add optional fields that might be present
    optional_fields = [
        "views",
        "forwards",
        "replies",
        "is_forwarded",
        "forwarded_from",
        "creator_username",
        "creator_first_name",
        "creator_last_name",
    ]

    for field in optional_fields:
        if field in msg and msg[field] is not None:
            cleaned_msg[field] = msg[field]

    # Remove fields that conflict with database schema
    fields_to_remove = [
        "id",
        "created_at",
        "updated_at",
        "deleted_at",
        "is_deleted",
        "text_length",
        "has_media",
    ]
    for field in fields_to_remove:
        cleaned_msg.pop(field, None)

    logger.debug(f"Message data before cleaning: {cleaned_msg}")

    # Clean pandas-specific values (NaN, NaT, etc.) and handle data types
    for key, value in cleaned_msg.items():
        if pd.isna(value) or (
            isinstance(value, str) and value in ["NaN", "NaT", "nan", "nat"]
        ):
            cleaned_msg[key] = None
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned_msg[key] = None
        elif isinstance(value, bool):
            # Convert boolean to string for API compatibility
            logger.debug(
                f"Converting boolean field '{key}' from {value} to '{str(value).lower()}'"
            )
            cleaned_msg[key] = str(value).lower()
        elif isinstance(value, str) and value.lower() in ["bool", "true", "false"]:
            # Handle string representations of boolean values
            if value.lower() == "bool":
                logger.debug(f"Converting string 'bool' field '{key}' to 'false'")
                cleaned_msg[key] = "false"  # Default to false for "bool" strings
            else:
                logger.debug(
                    f"Converting string boolean field '{key}' from '{value}' to '{value.lower()}'"
                )
                cleaned_msg[key] = value.lower()
        elif isinstance(value, list | dict):
            # Convert complex types to JSON strings using pandas
            try:
                # Use pandas to_json for complex objects
                if isinstance(value, dict):
                    cleaned_msg[key] = pd.DataFrame([value]).to_json(
                        orient="records", force_ascii=False, default_handler=str
                    )
                else:
                    cleaned_msg[key] = pd.Series(value).to_json(
                        force_ascii=False, default_handler=str
                    )
            except (TypeError, ValueError):
                cleaned_msg[key] = str(value)

    logger.debug(f"Message data after cleaning: {cleaned_msg}")

    # Convert float values to int for integer fields
    if (
        "file_size" in cleaned_msg
        and isinstance(cleaned_msg["file_size"], int | float)
        and cleaned_msg["file_size"] is not None
    ):
        try:
            cleaned_msg["file_size"] = int(cleaned_msg["file_size"])
        except (ValueError, TypeError):
            cleaned_msg["file_size"] = None

    # Convert numeric fields
    numeric_fields = ["views", "forwards", "replies"]
    for field in numeric_fields:
        if field in cleaned_msg and cleaned_msg[field] is not None:
            try:
                if isinstance(cleaned_msg[field], str) and cleaned_msg[field].isdigit():
                    cleaned_msg[field] = int(cleaned_msg[field])
                elif isinstance(cleaned_msg[field], int | float):
                    cleaned_msg[field] = int(cleaned_msg[field])
            except (ValueError, TypeError):
                cleaned_msg[field] = None

    logger.debug(f"Final cleaned message data: {cleaned_msg}")
    return cleaned_msg


async def process_import_batch(
    batch: list[dict[str, Any]],
    db_service: TelegramDBService,
    retry_handler: RetryHandler,
    skip_duplicates: bool = False,
) -> tuple[int, int, int]:
    """Process a batch of messages for import using individual endpoints."""
    logger.debug(
        f"Processing import batch of {len(batch)} messages using individual endpoints..."
    )
    logger.debug(f"Skip duplicates setting: {skip_duplicates}")

    success_count = 0
    error_count = 0
    skipped_count = 0

    # Process messages individually to avoid bulk endpoint data type issues
    for i, msg in enumerate(batch):
        try:
            if validate_message_format(msg):
                cleaned_msg = clean_message_data(msg)
                logger.debug(f"Final cleaned message data: {cleaned_msg}")

                # Use individual message import to avoid bulk endpoint issues
                result = await retry_handler.execute_with_retry(
                    db_service.store_message, cleaned_msg
                )

                if result:
                    success_count += 1
                    logger.debug(
                        f"Successfully imported message {i + 1}/{len(batch)}: {cleaned_msg.get('message_id')}"
                    )
                else:
                    error_count += 1
                    logger.warning(
                        f"Failed to import message {i + 1}/{len(batch)}: {cleaned_msg.get('message_id')}"
                    )
            else:
                skipped_count += 1
                logger.warning(f"Skipping invalid message {i + 1}/{len(batch)}")

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing message {i + 1}/{len(batch)}: {e}")
            continue

    logger.debug(
        f"Batch processing complete - Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}"
    )
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
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the import process."""
    stats: dict[str, Any] = {
        "total_messages": 0,
        "imported_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "start_time": datetime.now(),
        "end_time": None,
    }

    try:
        # Load messages from file using pandas
        logger.info(f"Loading messages from {data_file} using pandas...")
        messages = load_messages_from_file(data_file)
        stats["total_messages"] = len(messages)

        if not messages:
            logger.error("No messages found in file")
            return stats

        # Apply limit if specified
        if limit is not None:
            if limit <= 0:
                logger.error("Limit must be a positive number")
                return stats
            if limit > len(messages):
                logger.warning(
                    f"Limit ({limit}) is greater than available messages ({len(messages)}). Using all available messages."
                )
                limit = len(messages)

            original_count = len(messages)
            messages = messages[:limit]
            logger.info(
                f"Limited import to {len(messages)} messages (from {original_count} total)"
            )
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
                        quality_issues.extend(
                            [f"Message {i}: {issue}" for issue in issues]
                        )
                else:
                    invalid_count += 1

            logger.info(
                f"Validation complete: {valid_count} valid, {invalid_count} invalid messages"
            )

            if quality_issues:
                logger.warning(f"Found {len(quality_issues)} data quality issues:")
                for issue in quality_issues[:10]:  # Show first 10 issues
                    logger.warning(f"  {issue}")
                if len(quality_issues) > 10:
                    logger.warning(f"  ... and {len(quality_issues) - 10} more issues")

            return stats

        # Initialize database service
        logger.debug(f"Initializing database service with URL: {db_url}")
        async with TelegramDBService(db_url) as db_service:
            # Check connection
            logger.info("Checking database connection...")
            logger.debug("Sending connection test request...")
            if not await db_service.check_connection():
                logger.error(
                    "Cannot connect to database. Please check if the service is running."
                )
                return stats

            logger.info("Database connection successful")
            logger.debug("Database service initialized and connection verified")

            # Initialize retry handler
            retry_handler = RetryHandler(max_retries, retry_delay, max_delay)

            # Process messages in batches
            total_batches = (len(messages) + batch_size - 1) // batch_size
            logger.debug(
                f"Will process {len(messages)} messages in {total_batches} batches of size {batch_size}"
            )

            for i in range(0, len(messages), batch_size):
                batch = messages[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)"
                )
                logger.debug(
                    f"Batch {batch_num} message IDs: {[msg.get('message_id', 'unknown') for msg in batch[:5]]}{'...' if len(batch) > 5 else ''}"
                )

                if not dry_run:
                    logger.debug(f"Starting batch {batch_num} processing...")
                    success, errors, skipped = await process_import_batch(
                        batch, db_service, retry_handler, skip_duplicates
                    )

                    stats["imported_count"] += success
                    stats["error_count"] += errors
                    stats["skipped_count"] += skipped

                    logger.info(
                        f"Batch {batch_num} complete. Success: {success}, Errors: {errors}, Skipped: {skipped}"
                    )
                    logger.debug(
                        f"Batch {batch_num} statistics - Total: {len(batch)}, Success: {success}, Errors: {errors}, Skipped: {skipped}"
                    )

                    # Progress update
                    progress = min(i + batch_size, len(messages))
                    logger.info(f"Progress: {progress}/{len(messages)} messages")

                    # Add batch delay if not the last batch
                    if batch_num < total_batches and batch_delay > 0:
                        logger.debug(f"Waiting {batch_delay}s before next batch...")
                        await asyncio.sleep(batch_delay)
                else:
                    logger.info(
                        f"DRY RUN - Would process batch {batch_num} ({len(batch)} messages)"
                    )
                    logger.debug(
                        f"DRY RUN - Batch {batch_num} would contain: {[msg.get('message_id', 'unknown') for msg in batch[:5]]}{'...' if len(batch) > 5 else ''}"
                    )
                    stats["imported_count"] += len(batch)

            # Update retry statistics
            stats["retry_count"] = retry_handler.total_retries

    except Exception as e:
        logger.error(f"Import failed: {e}")
        stats["error_count"] += 1

    finally:
        stats["end_time"] = datetime.now()

    return stats


def import_single_file(file_path: str, db_url: str, logger: logging.Logger) -> bool:
    """Import messages from a single JSON file."""
    try:
        # Load and validate the JSON file
        logger.info("🔍 Loading and validating JSON file...")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate that this is in the expected structured DataFrame format
        if not (
            data.get("metadata", {}).get("data_format") == "structured_dataframe"
            and data.get("messages")
            and isinstance(data["messages"], list)
        ):
            logger.error("❌ File is not in the expected structured DataFrame format")
            logger.error("Expected: metadata.data_format = 'structured_dataframe' and messages array")
            return False
        
        logger.info(f"✅ JSON file validated - {len(data['messages'])} messages in structured format")
        
        # Run the import synchronously since we're not in an async context here
        import_result = asyncio.run(run_import(file_path, db_url))
        # Check if import was successful (import_result is a dict with stats)
        return import_result is not None and import_result.get("error_count", 0) == 0
        
    except Exception as e:
        logger.error(f"Failed to read or validate JSON file: {e}")
        return False


def import_all_combined_collections(db_url: str, logger: logging.Logger, verbose: bool = False) -> bool:
    """Import messages from all available combined collections."""
    try:
        collections_dir = Path(COLLECTIONS_DIR)
        if not collections_dir.exists():
            logger.error("❌ Collections directory not found")
            return False
        
        # Find all combined collection files
        combined_files = list(collections_dir.glob(COMBINED_COLLECTION_GLOB))
        if not combined_files:
            logger.warning("⚠️  No combined collection files found")
            return False
        
        logger.info(f"📁 Found {len(combined_files)} combined collection files")
        
        success_count = 0
        total_count = 0
        
        for file_path in combined_files:
            try:
                logger.info(f"🔄 Importing from: {file_path.name}")
                import_result = asyncio.run(run_import(str(file_path), db_url))
                if import_result:
                    success_count += 1
                    logger.info(f"✅ Successfully imported: {file_path.name}")
                else:
                    logger.error(f"❌ Failed to import: {file_path.name}")
                total_count += 1
            except Exception as e:
                logger.error(f"❌ Error importing {file_path.name}: {e}")
                total_count += 1
        
        logger.info(f"📊 Import Summary: {success_count}/{total_count} files imported successfully")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Failed to import combined collections: {e}")
        return False
