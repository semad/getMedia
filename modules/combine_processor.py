"""
Combine Processor Module

This module handles the combination of existing collection files into consolidated datasets.
It provides functionality to merge multiple collection files by channel and create
combined JSON files with metadata using pandas for JSON operations.
"""

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    RAW_COLLECTIONS_DIR,
    COLLECTIONS_DIR,
    COMBINED_DIR,
    F_PREFIX,
    COMBINED_FILE_PATTERN,
    RAW_COLLECTION_GLOB,
    ALL_JSON_GLOB,
)

logger = logging.getLogger(__name__)


def auto_detect_channels_from_raw(logger=None) -> list[str]:
    """
    Automatically detect channel names from existing collection files in the raw directory.

    Args:
        logger: Logger instance for output (optional)

    Returns:
        List of detected channel names
    """
    raw_dir = Path(RAW_COLLECTIONS_DIR)
    detected_channels = set()

    if not raw_dir.exists():
        if logger:
            logger.warning(f"Raw directory not found: {raw_dir}")
        return []

    # Find all JSON files in the raw directory
    json_files = list(raw_dir.glob(RAW_COLLECTION_GLOB))
    if not json_files:
        if logger:
            logger.warning("No collection files found in raw directory")
        return []

    if logger:
        logger.info(f"🔍 Scanning {len(json_files)} files in raw directory...")

    for file_path in json_files:
        try:
            # Extract channel name from filename pattern: {F_PREFIX}_{channel}_{...}.json
            filename = file_path.stem  # Remove .json extension
            parts = filename.split("_")
            if len(parts) >= 2:
                # The channel name is the second part after 'tg'
                channel_name = parts[1]
                detected_channels.add(channel_name)
                if logger:
                    logger.debug(
                        f"   📁 Detected channel: {channel_name} from {file_path.name}"
                    )
        except Exception as e:
            if logger:
                logger.debug(f"   ⚠️  Could not parse filename {file_path.name}: {e}")
            continue

    # Convert set to sorted list
    channel_list = sorted(detected_channels)
    if channel_list:
        if logger:
            logger.info(
                f"✅ Auto-detected {len(channel_list)} channels: {', '.join(channel_list)}"
            )
    else:
        if logger:
            logger.warning("⚠️  No channels could be detected from filenames")

    return channel_list


def auto_detect_channels_from_raw_advanced(raw_dir: Path) -> list[str]:
    """
    Auto-detect channels from raw collection files using advanced parsing.

    Args:
        raw_dir: Path to the raw collections directory

    Returns:
        List of detected channel names
    """
    if not raw_dir.exists():
        logger.warning(f"⚠️  Raw directory {raw_dir} does not exist")
        return []

    # Find all JSON files in raw directory
    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"⚠️  No JSON files found in {raw_dir}")
        return []

    channels = set()
    for file_path in json_files:
        filename = file_path.stem  # Get filename without extension

        # Parse filename: {F_PREFIX}_{channel}_{min_id}_{max_id}.json
        if filename.startswith(f"{F_PREFIX}_"):
            parts = filename.split("_")
            if len(parts) >= 4:  # tg + channel + min_id + max_id
                # Work backwards from the end to find the last two numeric parts (message IDs)
                # The channel name is everything between 'tg' and the last two parts
                try:
                    # Check if the last two parts are numeric (message IDs)
                    last_part = parts[-1]
                    second_last_part = parts[-2]

                    # Try to convert to int to verify they're message IDs
                    int(last_part)
                    int(second_last_part)

                    # If successful, everything between 'tg' and these two parts is the channel name
                    channel_parts = parts[
                        1:-2
                    ]  # Skip 'tg' and the last two numeric parts
                    if channel_parts:
                        channel_name = "_".join(channel_parts)
                        channels.add(channel_name)
                    else:
                        # Fallback: if no channel parts found
                        logger.warning(
                            f"⚠️  Could not extract channel name from {filename}"
                        )

                except (ValueError, IndexError):
                    # Fallback: try the old method
                    channel_parts = []
                    for _i, part in enumerate(parts[1:], 1):  # Skip 'tg'
                        try:
                            int(part)  # Try to convert to int
                            # If successful, we've found the first message ID
                            break
                        except ValueError:
                            # This is part of the channel name
                            channel_parts.append(part)

                    if channel_parts:
                        channel_name = "_".join(channel_parts)
                        channels.add(channel_name)
                    else:
                        # Final fallback: take the part after '{F_PREFIX}_' as channel name
                        if len(parts) > 1:
                            channels.add(parts[1])

    return sorted(channels)


def combine_existing_collections(
    channels: list[str] | None = None, verbose: bool = False
) -> dict[str, Any]:
    """
    Combine existing collection files by channel.

    This function ensures only one final combined file per channel across multiple calls.
    If a combined file already exists, it will merge new data with the existing data.

    Args:
        channels: List of channel names to combine. If None, auto-detect from raw files.
        verbose: Enable verbose logging

    Returns:
        Dictionary with results of the combine operation
    """
    if verbose:
        logger.info("🔧 Starting combine operation...")

    # Setup paths
    raw_dir = Path(RAW_COLLECTIONS_DIR)
    output_dir = Path(COMBINED_DIR)

    if not raw_dir.exists():
        logger.error(f"❌ Raw directory {raw_dir} does not exist")
        return {"success": False, "error": f"Raw directory {raw_dir} does not exist"}

    # Auto-detect channels if not provided
    if channels is None:
        channels = auto_detect_channels_from_raw(raw_dir)
        if verbose:
            logger.info(f"🔍 Auto-detected {len(channels)} channels: {channels}")

    if not channels:
        logger.error("❌ No channels specified or detected")
        return {"success": False, "error": "No channels specified or detected"}

    # Find all JSON files in raw directory
    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        logger.warning("⚠️  No existing collection files found to combine")
        return {
            "success": False,
            "error": "No existing collection files found to combine",
        }

    if verbose:
        logger.info(f"📁 Found {len(json_files)} JSON files in raw directory")

    results = {"success": True, "combined_files": [], "errors": [], "summary": {}}

    # Process each channel
    for channel in channels:
        if verbose:
            logger.info(f"🔄 Processing channel: {channel}")

        # Check if combined file already exists
        existing_combined_file = output_dir / COMBINED_FILE_PATTERN.format(channel=channel)
        existing_messages = []
        existing_metadata = {}

        if existing_combined_file.exists():
            if verbose:
                logger.info(
                    f"📂 Found existing combined file: {existing_combined_file.name}"
                )

            try:
                # Read existing combined file
                df = pd.read_json(existing_combined_file)
                if "messages" in df.columns:
                    existing_messages = (
                        df["messages"].iloc[0]
                        if len(df) == 1
                        else df["messages"].tolist()
                    )
                elif len(df) == 1 and "messages" in df.iloc[0]:
                    existing_messages = df.iloc[0]["messages"]
                    existing_metadata = df.iloc[0].get("metadata", {})
                else:
                    existing_messages = df.to_dict("records")

                if verbose:
                    logger.info(
                        f"📖 Loaded {len(existing_messages)} existing messages from combined file"
                    )

            except Exception as e:
                logger.warning(f"⚠️  Could not read existing combined file: {e}")
                existing_messages = []
                existing_metadata = {}

        # Find files for this channel
        channel_files = []
        for file_path in json_files:
            filename = file_path.stem
            # Parse filename to extract the actual channel name
            if filename.startswith(f"{F_PREFIX}_"):
                parts = filename.split("_")
                if len(parts) >= 4:  # tg + channel + min_id + max_id
                    try:
                        # Check if the last two parts are numeric (message IDs)
                        last_part = parts[-1]
                        second_last_part = parts[-2]

                        # Try to convert to int to verify they're message IDs
                        int(last_part)
                        int(second_last_part)

                        # Extract channel name from filename
                        file_channel_parts = parts[
                            1:-2
                        ]  # Skip 'tg' and the last two numeric parts
                        if file_channel_parts:
                            file_channel_name = "_".join(file_channel_parts)
                            # Only match if the extracted channel name exactly matches our target channel
                            if file_channel_name == channel:
                                channel_files.append(file_path)

                    except (ValueError, IndexError):
                        # Fallback: try simple prefix match but be more careful
                        if len(parts) > 1 and parts[1] == channel:
                            channel_files.append(file_path)

        if not channel_files:
            logger.warning(f"⚠️  No new files found for channel: {channel}")
            if existing_messages:
                # Keep existing combined file if no new files
                results["combined_files"].append(str(existing_combined_file))
                results["summary"][channel] = {
                    "source_files": 0,
                    "total_messages": len(existing_messages),
                    "output_file": str(existing_combined_file),
                    "status": "no_new_files",
                }
            else:
                results["errors"].append(f"No files found for channel: {channel}")
            continue

        if verbose:
            logger.info(
                f"📄 Found {len(channel_files)} new files for channel {channel}"
            )

        # Read and combine messages from all new files
        new_messages = []
        source_files = []
        message_id_ranges = []
        total_new_messages = 0

        for file_path in channel_files:
            try:
                # Use pandas to read JSON file with different strategies
                messages = []

                # Strategy 1: Try direct pandas read
                try:
                    df = pd.read_json(file_path)
                    if "messages" in df.columns:
                        messages = (
                            df["messages"].iloc[0]
                            if len(df) == 1
                            else df["messages"].tolist()
                        )
                    elif len(df) == 1 and "messages" in df.iloc[0]:
                        messages = df.iloc[0]["messages"]
                    else:
                        messages = df.to_dict("records")
                except Exception:
                    # Strategy 2: Try with lines=True for JSONL format
                    try:
                        df = pd.read_json(file_path, lines=True)
                        messages = df.to_dict("records")
                    except Exception:
                        # Strategy 3: Try reading as string first, then parse
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                content = f.read()
                            # Try to parse as JSON string using StringIO to avoid deprecation warning
                            json_content = (
                                f"[{content}]"
                                if not content.startswith("[")
                                else content
                            )
                            df = pd.read_json(StringIO(json_content))
                            if "messages" in df.columns:
                                messages = (
                                    df["messages"].iloc[0]
                                    if len(df) == 1
                                    else df["messages"].tolist()
                                )
                            elif len(df) == 1 and "messages" in df.iloc[0]:
                                messages = df.iloc[0]["messages"]
                            else:
                                messages = df.to_dict("records")
                        except Exception as e:
                            logger.error(
                                f"❌ All pandas strategies failed for {file_path.name}: {e}"
                            )
                            continue

                new_messages.extend(messages)
                source_files.append(file_path.name)

                # Extract message ID range from filename
                filename = file_path.stem
                if filename.startswith(f"{F_PREFIX}_{channel}_"):
                    parts = filename.split("_")
                    if len(parts) >= 4:
                        try:
                            min_id = int(parts[-2])
                            max_id = int(parts[-1])
                            message_id_ranges.append([min_id, max_id])
                        except ValueError:
                            pass

                total_new_messages += len(messages)

                if verbose:
                    logger.info(
                        f"  📖 Read {len(messages)} messages from {file_path.name}"
                    )

            except Exception as e:
                error_msg = f"Error reading {file_path.name}: {e}"
                logger.error(f"❌ {error_msg}")
                results["errors"].append(error_msg)

        # Combine existing and new messages
        all_messages = existing_messages + new_messages

        if not all_messages:
            logger.warning(f"⚠️  No messages found for channel: {channel}")
            results["errors"].append(f"No messages found for channel: {channel}")
            continue

        # Deduplicate messages by message_id (keep the latest occurrence)
        logger.info(f"🔍 Deduplicating messages for {channel}...")
        original_count = len(all_messages)
        seen_ids = set()
        deduplicated_messages = []
        
        # Process messages in reverse order to keep the latest occurrence
        for message in reversed(all_messages):
            message_id = message.get('message_id')
            if message_id not in seen_ids:
                seen_ids.add(message_id)
                deduplicated_messages.append(message)
        
        # Reverse back to original order
        deduplicated_messages.reverse()
        all_messages = deduplicated_messages
        
        duplicates_removed = original_count - len(all_messages)
        if duplicates_removed > 0:
            logger.info(f"✅ Removed {duplicates_removed} duplicate messages for {channel}")
        else:
            logger.info(f"✅ No duplicates found for {channel}")

        # Update or create metadata
        if existing_metadata:
            # Update existing metadata
            existing_source_files = existing_metadata.get("source_files", [])
            existing_ranges = existing_metadata.get("message_id_ranges", [])

            updated_metadata = {
                "combined_at": datetime.now().isoformat(),
                "channel": f"@{channel}",
                "source_files": existing_source_files + source_files,
                "message_id_ranges": existing_ranges + message_id_ranges,
                "total_messages": len(all_messages),
                "overall_range": f"{min([r[0] for r in (existing_ranges + message_id_ranges)]) if (existing_ranges + message_id_ranges) else 0}-{max([r[1] for r in (existing_ranges + message_id_ranges)]) if (existing_ranges + message_id_ranges) else 0}",
                "data_format": "combined_collection",
                "source_directory": "raw",
                "previous_combined_at": existing_metadata.get("combined_at"),
                "new_messages_added": total_new_messages,
                "existing_messages": len(existing_messages),
            }
        else:
            # Create new metadata
            updated_metadata = {
                "combined_at": datetime.now().isoformat(),
                "channel": f"@{channel}",
                "source_files": source_files,
                "message_id_ranges": message_id_ranges,
                "total_messages": len(all_messages),
                "overall_range": f"{min([r[0] for r in message_id_ranges]) if message_id_ranges else 0}-{max([r[1] for r in message_id_ranges]) if message_id_ranges else 0}",
                "data_format": "combined_collection",
                "source_directory": "raw",
                "new_messages_added": total_new_messages,
                "existing_messages": len(existing_messages),
            }

        # Create combined data
        combined_data = {"metadata": updated_metadata, "messages": all_messages}

        # Save combined file using pandas
        output_file = output_dir / COMBINED_FILE_PATTERN.format(channel=channel)
        try:
            # Convert combined data to pandas DataFrame for JSON export
            combined_df = pd.DataFrame([combined_data])
            combined_df.to_json(
                output_file, orient="records", indent=2, default_handler=str
            )

            results["combined_files"].append(str(output_file))
            results["summary"][channel] = {
                "source_files": len(source_files),
                "total_messages": len(all_messages),
                "new_messages": total_new_messages,
                "existing_messages": len(existing_messages),
                "output_file": str(output_file),
                "status": "updated" if existing_messages else "created",
            }

            if verbose:
                status = "updated" if existing_messages else "created"
                logger.info(
                    f"✅ {status.capitalize()} combined file with {len(all_messages)} total messages ({total_new_messages} new) in {output_file}"
                )

        except Exception as e:
            error_msg = f"Error saving combined file for {channel}: {e}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)

    # Log summary
    if verbose:
        logger.info("=" * 60)
        logger.info("📋 COMBINE OPERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"✅ Successfully processed: {len(results['combined_files'])} files"
        )
        if results["errors"]:
            logger.info(f"❌ Errors: {len(results['errors'])}")
            for error in results["errors"]:
                logger.info(f"  - {error}")
        logger.info("=" * 60)

    return results


def get_combine_help_text() -> str:
    """Get help text for the combine command."""
    return """
Combine existing collection files into consolidated datasets.

This command reads all JSON files from the 'reports/collections/raw' directory
and combines them by channel into consolidated files in 'reports/collections'
using pandas for all JSON operations.

IMPORTANT: This command creates CUMULATIVE combined files. Each run will:
- Check for existing combined files
- Merge new data with existing data
- Maintain only ONE final combined file per channel across multiple calls

Features:
- Auto-detects channels from filenames
- Preserves message metadata and structure
- Creates comprehensive metadata for each combined file
- Handles multiple source files per channel
- Uses pandas exclusively for JSON processing with multiple parsing strategies
- Maintains cumulative data across multiple combine operations

Usage:
  python main.py combine                    # Auto-detect and combine all channels
  python main.py combine --channels books   # Combine specific channels
  python main.py combine --channels books magazines --verbose  # Multiple channels with verbose output

Output:
- Combined files saved to: {COLLECTIONS_DIR}/{COMBINED_FILE_PATTERN}
- Each file contains ALL messages for that channel (cumulative across all combine operations)
- Metadata includes source files, message ranges, combination timestamp, and update history
- JSON files are processed using pandas with multiple parsing strategies for maximum compatibility
- Subsequent runs will update existing files rather than create new ones
"""
