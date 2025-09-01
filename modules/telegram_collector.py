"""
Telegram message collector for gathering messages from channels.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import Message

from .models import ChannelConfig, RateLimitConfig

# Constants for file naming - now imported from config
from config import (
    RAW_COLLECTIONS_DIR,
    F_PREFIX,
    F_SEPARATOR,
    F_EXTENSION,
)

logger = logging.getLogger(__name__)


def convert_messages_to_dataframe(messages: list[Any]) -> pd.DataFrame:
    """Convert TelegramMessage objects to a pandas DataFrame."""
    if not messages:
        return pd.DataFrame()

    # Prepare data for DataFrame
    data_rows = []

    for msg in messages:
        try:
            if hasattr(msg, "__dict__"):
                # Handle TelegramMessage objects
                msg_dict = {}
                for attr_name, attr_value in msg.__dict__.items():
                    if hasattr(attr_value, "isoformat"):
                        msg_dict[attr_name] = attr_value.isoformat()
                    else:
                        msg_dict[attr_name] = attr_value
                data_rows.append(msg_dict)
            elif isinstance(msg, dict):
                # Handle already-dict messages
                msg_dict = {}
                for key, value in msg.items():
                    if hasattr(value, "isoformat"):
                        msg_dict[key] = value.isoformat()
                    else:
                        msg_dict[key] = value
                data_rows.append(msg_dict)
            else:
                logger.warning(f"Message has unexpected format: {type(msg)}")
                continue
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue

    if data_rows:
        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Clean up the DataFrame
        df = df.replace([pd.NA, pd.NaT, "NaN", "NaT", "nan", "nat"], None)

        # Convert numeric columns
        numeric_columns = ["message_id", "file_size", "views", "forwards", "replies"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert date columns
        date_columns = ["date", "edit_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info(f"🔧 Converted {len(df)} messages to pandas DataFrame")
        logger.info(f"📋 DataFrame shape: {df.shape}")
        logger.info(f"📋 DataFrame columns: {list(df.columns)}")

        return df
    else:
        logger.warning("No valid messages to convert to DataFrame")
        return pd.DataFrame()


def export_messages_to_file(
    messages: list[Any], export_path: str, channel_list: list[Any]
) -> str | None:
    """Export messages directly to JSON using pandas DataFrame."""
    try:
        # Ensure export directory exists
        os.makedirs(RAW_COLLECTIONS_DIR, exist_ok=True)

        # Generate filename
        channel_names = [
            ch.username.replace("@", "") for ch in channel_list if ch.enabled
        ]
        channel_str = F_SEPARATOR.join(channel_names) if channel_names else "unknown"

        # Get message ID range
        if messages:
            message_ids = [
                msg.message_id
                if hasattr(msg, "message_id")
                else (msg.get("message_id") if isinstance(msg, dict) else None)
                for msg in messages
            ]
            message_ids = [mid for mid in message_ids if mid is not None]
            if message_ids:
                min_id = min(message_ids)
                max_id = max(message_ids)
                # Always use raw directory for new collections
                filename = f"{RAW_COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}{min_id}{F_SEPARATOR}{max_id}{F_EXTENSION}"
            else:
                # Use raw directory for unknown message IDs
                filename = f"{RAW_COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}unknown{F_SEPARATOR}unknown{F_EXTENSION}"
        else:
            # Use raw directory for no messages
            filename = f"{RAW_COLLECTIONS_DIR}/{F_PREFIX}{F_SEPARATOR}{channel_str}{F_SEPARATOR}none{F_SEPARATOR}none{F_EXTENSION}"

        # Convert messages directly to pandas DataFrame
        df = convert_messages_to_dataframe(messages)

        if df.empty:
            logger.warning("No messages to export")
            return None

        # Create a simplified export structure that avoids complex object issues
        # Convert DataFrame to a list of dictionaries with safe string conversion
        messages_list = []
        for _, row in df.iterrows():
            message_dict = {}
            for column in df.columns:
                try:
                    value = row[column]
                    if value is None:
                        message_dict[column] = None
                    elif isinstance(value, (str, int, float, bool)):
                        message_dict[column] = value
                    else:
                        # Convert complex objects to string representation
                        message_dict[column] = str(value)
                except Exception as e:
                    # If conversion fails, use a safe fallback
                    message_dict[column] = f"[Error converting {column}: {e}]"
            messages_list.append(message_dict)

        # Create structured export data
        export_data = {
            "metadata": {
                "collected_at": datetime.now().isoformat(),
                "channels": [ch.username for ch in channel_list if ch.enabled],
                "data_format": "structured_dataframe",
                "total_messages": len(df),
                "dataframe_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {
                        col: str(dtype) for col, dtype in df.dtypes.to_dict().items()
                    },
                },
            },
            "messages": messages_list,
        }

        # Export directly using pandas DataFrame
        try:
            # Create DataFrame with the export structure and export directly
            export_df = pd.DataFrame([export_data])
            export_df.to_json(
                filename,
                orient="records",
                indent=2,
                default_handler=str,
                force_ascii=False,
            )
        except Exception as e:
            logger.error(f"Pandas JSON export failed: {e}")
            # Fallback: use json module with proper encoding handling
            import json

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            except Exception as json_error:
                logger.error(f"JSON fallback also failed: {json_error}")
                # Final fallback: write raw data as string with error handling
                try:
                    with open(filename, "w", encoding="utf-8", errors="replace") as f:
                        f.write(str(export_data))
                except Exception as final_error:
                    logger.error(f"All export methods failed: {final_error}")
                    return None

        logger.info(f"✅ Successfully exported {len(df)} messages to: {filename}")
        logger.info(f"📊 DataFrame shape: {df.shape}")
        logger.info(f"📋 Columns: {list(df.columns)}")

        return filename

    except Exception as e:
        logger.error(f"❌ Failed to export messages: {e}")
        return None


class TelegramCollector:
    """Collects messages from Telegram channels and exports them to files."""

    def __init__(self, rate_config: RateLimitConfig):
        self.client: TelegramClient | None = None
        self.rate_config = rate_config
        self.stats = {"total_messages": 0, "channels_processed": 0, "errors": 0}
        self.collected_messages: list[dict[str, Any]] = []

    async def initialize(self, api_id: str, api_hash: str, session_name: str) -> bool:
        """Initialize the Telegram client."""
        try:
            # Validate and convert API ID
            try:
                api_id_int = int(api_id)
            except ValueError:
                logger.error(f"Invalid API ID: {api_id}. Must be a valid integer.")
                return False

            self.client = TelegramClient(session_name, api_id_int, api_hash)
            await self.client.start()
            logger.info("Telegram client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            return False

    async def collect_from_channel(
        self,
        channel_config: ChannelConfig,
        max_messages: int | None = None,
        offset_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Collect messages from a specific channel."""
        if not self.client:
            logger.error("Telegram client not initialized")
            return []

        messages = []
        message_count = 0

        # Set max_messages if not specified
        if max_messages == 0:
            max_messages = channel_config.max_messages_per_session

        try:
            logger.info(f"Starting collection from channel: {channel_config.username}")

            # Prepare collection parameters
            collection_params = {}
            if max_messages is not None:
                collection_params["limit"] = max_messages
            if offset_id is not None and offset_id > 0:
                collection_params["max_id"] = offset_id

            # Collect messages
            async for message in self.client.iter_messages(
                channel_config.username, **collection_params
            ):
                # Optimized rate limiting - only apply every 10 messages for high rates
                if (
                    message_count % 10 == 0
                    and self.rate_config.messages_per_minute > 1000
                ):
                    await self._wait_for_rate_limit()

                # Process message
                telegram_message = await self.process_message(
                    message, channel_config.username
                )
                if telegram_message:
                    messages.append(telegram_message)
                    self.collected_messages.append(telegram_message)
                    message_count += 1
                    self.stats["total_messages"] += 1

                # Progress update
                if message_count % 10 == 0:
                    logger.info(
                        f"Processed {message_count} messages from {channel_config.username}"
                    )

                # Check if we've reached the limit
                if max_messages is not None and message_count >= max_messages:
                    break

            logger.info(
                f"Completed collection from {channel_config.username}: {message_count} messages"
            )
            self.stats["channels_processed"] += 1

        except Exception as e:
            logger.error(
                f"Error collecting from channel {channel_config.username}: {e}"
            )
            self.stats["errors"] += 1

        return messages

    async def process_message(
        self, message: Message, channel_username: str
    ) -> dict[str, Any] | None:
        """Process a Telegram message and extract metadata."""
        try:
            # Validate message has required fields
            if not hasattr(message, "id") or message.id is None:
                return None

            # Get message date
            message_date = (
                message.date
                if hasattr(message, "date") and message.date
                else datetime.now()
            )

            # Extract creator/sender information
            creator_info = self._extract_creator_info(message)

            # Extract media information
            media_info = self._extract_media_info(message)

            # Extract text content
            message_text = self._extract_text_content(message)

            # Extract engagement metrics
            engagement_metrics = self._extract_engagement_metrics(message)

            # Extract only the fields needed by the database
            # Create message dictionary with only required and optional fields
            telegram_message = {
                "message_id": message.id,
                "channel_username": channel_username,
                "date": message_date,
                "text": message_text,
                **creator_info,
                **media_info,
                **engagement_metrics,
            }

            return telegram_message

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None

    def _extract_media_info(self, message: Message) -> dict[str, Any]:
        """Extract media information from a message."""
        media_type = None
        file_name = None
        file_size = None
        mime_type = None
        caption = None

        # Extract caption if present
        if hasattr(message, "caption") and message.caption:
            caption = str(message.caption)

        if hasattr(message, "media") and message.media:
            # Check if it's a document
            if hasattr(message.media, "document") and message.media.document:
                media_type = "document"
                if (
                    hasattr(message.media.document, "attributes")
                    and message.media.document.attributes
                ):
                    for attr in message.media.document.attributes:
                        if hasattr(attr, "file_name") and attr.file_name:
                            file_name = attr.file_name
                            break
                file_size = getattr(message.media.document, "size", None)
                mime_type = getattr(message.media.document, "mime_type", None)

            # Check if it's a photo
            elif hasattr(message.media, "photo") and message.media.photo:
                media_type = "photo"
                file_size = 0

        return {
            "media_type": media_type,
            "file_name": file_name,
            "file_size": file_size,
            "mime_type": mime_type,
            "caption": caption,
        }

    def _extract_text_content(self, message: Message) -> str:
        """Extract text content from a message."""
        message_text = ""
        if hasattr(message, "text") and message.text:
            message_text = str(message.text)

        if not message_text or len(message_text.strip()) == 0:
            message_text = "[No text content]"

        return message_text

    def _extract_engagement_metrics(self, message: Message) -> dict[str, Any]:
        """Extract engagement metrics from a message."""
        replies_count = None
        views_count = None
        forwards_count = None
        is_forwarded = False
        forwarded_from = None

        # Check if message is forwarded
        if hasattr(message, "forward") and message.forward is not None:
            is_forwarded = True
            # Try to get the source channel username
            if hasattr(message.forward, "chat") and message.forward.chat:
                forwarded_from = getattr(message.forward.chat, "username", None)

        if hasattr(message, "replies") and message.replies:
            replies_count = message.replies.replies

        if hasattr(message, "views"):
            views_count = message.views

        if hasattr(message, "forwards"):
            forwards_count = message.forwards

        return {
            "views": views_count,
            "forwards": forwards_count,
            "replies": replies_count,
            "is_forwarded": is_forwarded,
            "forwarded_from": forwarded_from,
        }

    def _extract_creator_info(self, message: Message) -> dict[str, Any]:
        """Extract creator/sender information from a message."""
        creator_username = None
        creator_first_name = None
        creator_last_name = None

        # Try to get sender information
        if hasattr(message, "sender") and message.sender:
            sender = message.sender
            creator_username = getattr(sender, "username", None)
            creator_first_name = getattr(sender, "first_name", None)
            creator_last_name = getattr(sender, "last_name", None)

        # Fallback to sender_id if available
        if not creator_username and hasattr(message, "sender_id"):
            creator_username = f"user_{message.sender_id}"

        return {
            "creator_username": creator_username,
            "creator_first_name": creator_first_name,
            "creator_last_name": creator_last_name,
        }

    async def _wait_for_rate_limit(self):
        """Wait for rate limiting."""
        # Optimized rate limiting - only wait if we're hitting limits
        # With 5000 messages per minute, we can process ~83 messages per second
        # Only add delay if we're processing very fast
        if self.rate_config.messages_per_minute > 1000:
            # For high rates, use minimal delay
            await asyncio.sleep(0.01)  # 10ms delay instead of full calculation
        else:
            # For lower rates, use standard calculation
            await asyncio.sleep(60 / self.rate_config.messages_per_minute)

    def get_collected_messages(self) -> list[dict[str, Any]]:
        """Get all messages collected so far."""
        return self.collected_messages.copy()

    def clear_collected_messages(self):
        """Clear the collected messages list."""
        self.collected_messages.clear()

    def export_messages_to_file(
        self, export_path: str, channel_list: list[ChannelConfig]
    ) -> str | None:
        """Export collected messages to structured file (JSON with DataFrame-like structure)."""
        return export_messages_to_file(
            self.collected_messages, export_path, channel_list
        )

    async def close(self):
        """Close the Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")
