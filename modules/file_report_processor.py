"""
Report processing module for Telegram data analysis.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    COLLECTIONS_DIR,
    COMBINED_JSON_GLOB,
    FILES_CHANNELS_DIR,
    REPORT_FILE_PATTERN,
    SUMMARY_FILE_PATTERN,
)

from .report_generator import generate_pandas_report

logger = logging.getLogger(__name__)


def parse_json_file(file_path: Path) -> list[dict] | None:
    """
    Parse JSON file with multiple fallback strategies.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of message dictionaries or None if parsing fails
    """
    strategies = [
        _parse_direct_pandas,
        _parse_jsonl_format,
        _parse_string_content,
    ]

    for strategy in strategies:
        try:
            result = strategy(file_path)
            if result is not None:
                return result
        except Exception as e:
            logger.debug(f"Strategy {strategy.__name__} failed: {e}")
            continue

    logger.error(f"All parsing strategies failed for {file_path}")
    return None


def _parse_direct_pandas(file_path: Path) -> list[dict] | None:
    """Strategy 1: Direct pandas JSON parsing."""
    df = pd.read_json(file_path)

    if "messages" in df.columns:
        if len(df) == 1:
            return df["messages"].iloc[0]
        return df["messages"].tolist()
    elif len(df) == 1 and "messages" in df.iloc[0]:
        return df.iloc[0]["messages"]
    else:
        return df.to_dict("records")


def _parse_jsonl_format(file_path: Path) -> list[dict] | None:
    """Strategy 2: JSONL format parsing."""
    df = pd.read_json(file_path, lines=True)
    return df.to_dict("records")


def _parse_string_content(file_path: Path) -> list[dict] | None:
    """Strategy 3: String content parsing with fallback."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Ensure content is wrapped in array brackets
    if not content.startswith("["):
        content = f"[{content}]"

    from io import StringIO

    df = pd.read_json(StringIO(content))

    if "messages" in df.columns:
        if len(df) == 1:
            return df["messages"].iloc[0]
        return df["messages"].tolist()
    elif len(df) == 1 and "messages" in df.iloc[0]:
        return df.iloc[0]["messages"]
    else:
        return df.to_dict("records")


def process_channel_reports(
    collections_dir: str = COLLECTIONS_DIR,
    output_dir: str = FILES_CHANNELS_DIR,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Process all channel reports from combined JSON files.

    Args:
        collections_dir: Directory containing combined JSON files
        output_dir: Output directory for generated reports
        verbose: Enable verbose logging

    Returns:
        Dictionary with processing results
    """
    # Use default formats
    formats = ["json", "txt"]

    # Set up logging
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all combined JSON files
    collections_path = Path(collections_dir)
    json_files = list(collections_path.glob(COMBINED_JSON_GLOB))

    if not json_files:
        logger.warning(f"No combined JSON files found in {collections_dir}")
        return {"error": "No combined JSON files found"}

    logger.info(f"📁 Found {len(json_files)} JSON files in collections directory")

    results = {
        "total_channels": len(json_files),
        "successful": 0,
        "failed": 0,
        "errors": [],
        "files_generated": [],
    }

    for json_file in json_files:
        try:
            # Extract channel name from filename
            channel_name = json_file.stem.replace("_combined", "")
            logger.info(f"🔄 Processing: {channel_name}")

            # Parse JSON file
            messages = parse_json_file(json_file)
            if messages is None:
                results["failed"] += 1
                results["errors"].append(f"Failed to parse {channel_name}")
                continue

            # Generate report
            df = pd.DataFrame(messages)
            report = generate_pandas_report(df, channel_name)

            # Save report files
            saved_files = _save_report_files(channel_name, report, output_path, formats)

            results["successful"] += 1
            results["files_generated"].extend(saved_files)

            logger.info(
                f"✅ Report generated for {channel_name}: {len(saved_files)} files saved"
            )

        except Exception as e:
            error_msg = f"Error processing {channel_name}: {e}"
            logger.error(error_msg)
            results["failed"] += 1
            results["errors"].append(error_msg)

    return results


def _save_report_files(
    channel_name: str, report: dict, output_path: Path, formats: list[str]
) -> list[str]:
    """Save report files in specified formats."""
    saved_files = []

    # Save JSON report
    if "json" in formats:
        json_file_path = output_path / REPORT_FILE_PATTERN.format(channel=channel_name)
        pd.DataFrame([report]).to_json(
            json_file_path, orient="records", indent=2, default_handler=str
        )
        saved_files.append(str(json_file_path))

    # Save summary text file
    if "txt" in formats or "summary" in formats:
        summary_text = generate_summary_text(report, channel_name)
        summary_file_path = output_path / SUMMARY_FILE_PATTERN.format(
            channel=channel_name
        )
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        saved_files.append(str(summary_file_path))

    return saved_files


def generate_summary_text(report: dict[str, Any], channel_name: str) -> str:
    """Generate a human-readable summary text file."""
    lines = []

    # Header
    lines.append(f"Channel Report Summary: {channel_name}")
    lines.append("=" * 50)
    lines.append("")

    # Basic statistics
    lines.append(f"Total Messages: {report.get('total_messages', 'N/A')}")
    lines.append(f"Total Columns: {report.get('total_columns', 'N/A')}")
    lines.append(f"Dataframe Shape: {report.get('dataframe_shape', 'N/A')}")
    lines.append("")

    # Media statistics
    lines.append(f"Media Messages: {report.get('media_messages', 'N/A')}")
    lines.append(f"Text Messages: {report.get('text_messages', 'N/A')}")
    lines.append(f"Total File Size: {report.get('total_file_size', 'N/A')}")
    lines.append("")

    # Date range
    lines.append(f"Date Range: {report.get('date_range', 'N/A')}")
    lines.append(f"Active Days: {report.get('active_days', 'N/A')}")
    lines.append("")

    # Field analysis
    lines.append("Field Analysis:")
    lines.append("-" * 30)
    lines.append("")

    field_analysis = report.get("field_analysis", {})
    for field_name, field_info in field_analysis.items():
        lines.append(f"{field_name}:")
        lines.append(f"  Data Type: {field_info.get('data_type', 'N/A')}")
        lines.append(f"  Non-null Values: {field_info.get('non_null_count', 'N/A')}")
        lines.append(
            f"  Null Values: {field_info.get('null_count', 'N/A')} ({field_info.get('null_percentage', 'N/A')}%)"
        )

        # Add unique values for categorical fields
        unique_count = field_info.get("unique_count", 0)
        if unique_count > 0 and unique_count <= 20:
            lines.append(f"  Unique Values: {unique_count}")
            unique_values = field_info.get("unique_values", [])
            if unique_values:
                lines.append(
                    f"  Sample Values: {', '.join(map(str, unique_values[:5]))}"
                )

        lines.append("")

    return "\n".join(lines)


def display_results_summary(results: dict[str, Any]):
    """Display a summary of the report generation results."""
    logger.info("=" * 60)
    logger.info("📋 CHANNEL REPORT GENERATION SUMMARY")
    logger.info("=" * 60)

    total_channels = results.get("total_channels", 0)
    successful = results.get("successful", 0)
    failed = results.get("failed", 0)
    errors = results.get("errors", [])

    logger.info(f"✅ Total Channels: {total_channels}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")

    if errors:
        logger.info("❌ Errors:")
        for error in errors:
            logger.info(f"   - {error}")

    logger.info("=" * 60)
    logger.info(
        f"📈 SUMMARY: {successful} successful, {failed} failed, {len(errors)} errors"
    )
    logger.info(f"📁 All reports saved to: {FILES_CHANNELS_DIR}")
    logger.info("🎉 Channel reports generated successfully!")
    logger.info("💡 Use the generated files for further analysis and visualization")
