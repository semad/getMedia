"""
Database Report Processor Module

This module handles report generation from database API endpoints.
It fetches data from the database, processes it with pandas, and generates
comprehensive reports saved to DB_CHANNELS_DIR.
"""

import logging
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from config import (
    API_ENDPOINTS,
    DB_CHANNELS_DIR,
    DEFAULT_DB_URL,
    REPORT_FILE_PATTERN,
    SUMMARY_FILE_PATTERN,
)

from .report_generator import generate_pandas_report

logger = logging.getLogger(__name__)


class DatabaseReportProcessor:
    """Processes reports from database API endpoints."""

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        self.db_url = db_url.rstrip("/")
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def process_channel_reports_from_db(
        self,
        channels: list[str] | None = None,
        output_dir: str = DB_CHANNELS_DIR,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Process channel reports from database API endpoints.

        Args:
            channels: List of channel usernames to process. If None, auto-detect.
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

        # Get channels to process
        if channels is None:
            channels = await self._get_available_channels()
            if not channels:
                logger.warning("No channels found in database")
                return {"error": "No channels found in database"}

        logger.info(f"📊 Processing {len(channels)} channels from database...")

        results = {
            "total_channels": len(channels),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "files_generated": [],
        }

        for channel in channels:
            try:
                logger.info(f"🔄 Processing channel: {channel}")

                # Fetch channel data from database
                messages = await self._fetch_channel_messages(channel)
                if not messages:
                    logger.warning(f"No messages found for channel: {channel}")
                    results["failed"] += 1
                    results["errors"].append(f"No messages found for {channel}")
                    continue

                # Convert to DataFrame and generate report
                df = pd.DataFrame(messages)
                report = generate_pandas_report(df, channel)

                # Save report files
                saved_files = self._save_report_files(
                    channel, report, df, output_path, formats
                )

                results["successful"] += 1
                results["files_generated"].extend(saved_files)

                logger.info(
                    f"✅ Report generated for {channel}: {len(saved_files)} files saved"
                )

            except Exception as e:
                error_msg = f"Error processing {channel}: {e}"
                logger.error(error_msg)
                results["failed"] += 1
                results["errors"].append(error_msg)

        return results

    async def _get_available_channels(self) -> list[str]:
        """Get list of available channels from database."""
        try:
            url = f"{self.db_url}{API_ENDPOINTS['channels']}"
            if self.session:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        channels_data = await response.json()
                        # Extract channel usernames
                        if isinstance(channels_data, list):
                            return [
                                ch.get("username", "")
                                for ch in channels_data
                                if ch.get("username")
                            ]
                        elif (
                            isinstance(channels_data, dict)
                            and "channels" in channels_data
                        ):
                            return [
                                ch.get("username", "")
                                for ch in channels_data["channels"]
                                if ch.get("username")
                            ]
                        else:
                            logger.warning("Unexpected channels data format")
                            return []
                    else:
                        logger.error(f"Failed to fetch channels: {response.status}")
                        return []
            else:
                logger.error("No database session available")
                return []
        except Exception as e:
            logger.error(f"Error fetching channels: {e}")
            return []

    async def _fetch_channel_messages(self, channel: str) -> list[dict[str, Any]]:
        """Fetch all messages for a specific channel from database."""
        try:
            url = f"{self.db_url}{API_ENDPOINTS['export_all']}"
            params = {
                "fields": "message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at",
                "channel": channel,
            }

            if self.session:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Filter for the specific channel
                        channel_messages = [
                            msg
                            for msg in result
                            if msg.get("channel_username") == channel
                        ]
                        return channel_messages
                    else:
                        logger.error(
                            f"API request failed for {channel}: {response.status}"
                        )
                        return []
            else:
                logger.error("No database session available")
                return []
        except Exception as e:
            logger.error(f"Error fetching messages for {channel}: {e}")
            return []

    def _save_report_files(
        self,
        channel: str,
        report: dict[str, Any],
        df: pd.DataFrame,
        output_path: Path,
        formats: list[str],
    ) -> list[str]:
        """Save report files in specified formats."""
        saved_files = []

        # Save JSON report
        if "json" in formats:
            json_file_path = output_path / REPORT_FILE_PATTERN.format(channel=channel)
            pd.DataFrame([report]).to_json(
                json_file_path, orient="records", indent=2, default_handler=str
            )
            saved_files.append(str(json_file_path))

        # Save summary text file
        if "txt" in formats or "summary" in formats:
            summary_text = self._generate_summary_text(report, channel)
            summary_file_path = output_path / SUMMARY_FILE_PATTERN.format(
                channel=channel
            )
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            saved_files.append(str(summary_file_path))

        # Save CSV export
        if "csv" in formats:
            csv_file_path = output_path / f"{channel}_data.csv"
            df.to_csv(csv_file_path, index=False, encoding="utf-8")
            saved_files.append(str(csv_file_path))

        # Save Excel export
        if "excel" in formats or "xlsx" in formats:
            excel_file_path = output_path / f"{channel}_data.xlsx"
            df.to_excel(excel_file_path, index=False, engine="openpyxl")
            saved_files.append(str(excel_file_path))

        return saved_files

    def _generate_summary_text(self, report: dict[str, Any], channel: str) -> str:
        """Generate a human-readable summary text file."""
        lines = []

        # Header
        lines.append(f"Database Channel Report Summary: {channel}")
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
            lines.append(
                f"  Non-null Values: {field_info.get('non_null_count', 'N/A')}"
            )
            lines.append(
                f"  Null Values: {field_info.get('null_count', 'N/A')} ({field_info.get('null_percentage', 'N/A')}%)"
            )
            lines.append(f"  Unique Values: {field_info.get('unique_count', 'N/A')}")
            lines.append("")

        return "\n".join(lines)


def display_db_results_summary(results: dict[str, Any]):
    """Display a summary of the database report generation results."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("📋 DATABASE CHANNEL REPORT GENERATION SUMMARY")
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
    logger.info(f"📁 All reports saved to: {DB_CHANNELS_DIR}")
    logger.info("🎉 Database channel reports generated successfully!")
    logger.info("💡 Use the generated files for further analysis and visualization")
