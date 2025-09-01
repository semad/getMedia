"""
Channel Reporter for Telegram message analysis.

This service generates comprehensive pandas-based reports for each channel:
- Detailed statistics and metrics
- Time series analysis
- Content analysis
- Export to various formats (CSV, JSON, Excel)
- Ready for visualization
- Uses pandas for all data conversions and serialization
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add the current directory to the Python path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    API_ENDPOINTS,
    FILES_CHANNELS_DIR,
    REPORT_FILE_PATTERN,
    SUMMARY_FILE_PATTERN,
)
from modules.database_service import TelegramDBService


class ChannelReporter:
    """Generates comprehensive reports for individual Telegram channels using pandas."""

    def __init__(
        self, db_service: TelegramDBService, output_dir: str = FILES_CHANNELS_DIR
    ):
        self.db_service = db_service
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    async def generate_channel_report(
        self, channel: str, formats: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Generate comprehensive report for a specific channel."""
        try:
            self.logger.info(
                f"📊 Generating comprehensive report for channel: {channel}"
            )

            # Fetch channel data
            async with self.db_service:
                # Get all messages for this channel
                messages = await self._fetch_channel_messages(channel)

                if not messages:
                    self.logger.warning(f"No messages found for channel: {channel}")
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(messages)
                self.logger.info(f"✅ Loaded {len(df):,} messages for {channel}")

                # Generate comprehensive report using pandas
                report = self._create_channel_report_pandas(channel, df)

                # Save report to files (using specified formats or JSON only by default)
                saved_files = await self._save_channel_report_pandas(
                    channel, report, df, formats
                )

                report["saved_files"] = saved_files
                return report

        except Exception as e:
            self.logger.error(f"Error generating report for channel {channel}: {e}")
            return None

    async def _fetch_channel_messages(self, channel: str) -> list[dict[str, Any]]:
        """Fetch all messages for a specific channel."""
        try:
            # Use the bulk export endpoint for efficiency
            url = f"{self.db_service.db_url}{API_ENDPOINTS['export_all']}"
            params = {
                "fields": "message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at",
                "channel": channel,
            }

            if self.db_service.session:
                async with self.db_service.session.get(url, params=params) as response:
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
                        self.logger.error(
                            f"API request failed with status {response.status}"
                        )
                        return []
            else:
                self.logger.error("No database session available")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching messages for {channel}: {e}")
            return []

    def _create_channel_report_pandas(
        self, channel: str, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Create a comprehensive channel report using pandas analysis."""
        try:
            # Basic statistics
            total_messages = len(df)

            # Field analysis
            field_analysis = {}
            for column in df.columns:
                col_data = df[column]
                field_info = {
                    "data_type": str(col_data.dtype),
                    "non_null_count": col_data.count(),
                    "null_count": col_data.isnull().sum(),
                    "null_percentage": round(
                        (col_data.isnull().sum() / len(col_data)) * 100, 2
                    ),
                    "unique_count": col_data.nunique(),
                }
                field_analysis[column] = field_info

            # Media analysis
            if "media_type" in df.columns:
                media_messages = len(
                    df[df["media_type"].notna() & (df["media_type"] != "")]
                )
                media_types = df["media_type"].value_counts().to_dict()
            else:
                media_messages = 0
                media_types = {}

            text_messages = total_messages - media_messages

            # Date analysis
            if "date" in df.columns:
                try:
                    date_col = pd.to_datetime(df["date"], errors="coerce")
                    valid_dates = date_col.dropna()
                    if len(valid_dates) > 0:
                        date_range = f"{valid_dates.min()} to {valid_dates.max()}"
                        active_days = valid_dates.dt.date.nunique()
                    else:
                        date_range = "No valid dates found"
                        active_days = 0
                except Exception:
                    date_range = "Date parsing failed"
                    active_days = 0
            else:
                date_range = "No date column found"
                active_days = 0

            # File size analysis
            if "file_size" in df.columns:
                try:
                    file_sizes = pd.to_numeric(df["file_size"], errors="coerce")
                    valid_sizes = file_sizes.dropna()
                    if len(valid_sizes) > 0:
                        total_file_size = valid_sizes.sum()
                        avg_file_size = valid_sizes.mean()
                    else:
                        total_file_size = avg_file_size = 0
                except Exception:
                    total_file_size = avg_file_size = 0
            else:
                total_file_size = avg_file_size = 0

            # Create comprehensive report
            report = {
                "channel_name": channel,
                "generated_at": datetime.now().isoformat(),
                "total_messages": total_messages,
                "total_columns": len(df.columns),
                "dataframe_shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
                "media_messages": media_messages,
                "text_messages": text_messages,
                "total_file_size": total_file_size,
                "average_file_size": avg_file_size,
                "date_range": date_range,
                "active_days": active_days,
                "field_analysis": field_analysis,
                "media_types": media_types,
            }

            return report

        except Exception as e:
            self.logger.error(f"Error creating report for {channel}: {e}")
            return {
                "channel_name": channel,
                "error": str(e),
                "total_messages": len(df) if df is not None else 0,
            }

    async def _save_channel_report_pandas(
        self,
        channel: str,
        report: dict[str, Any],
        df: pd.DataFrame,
        formats: list[str] | None = None,
    ) -> list[str]:
        """Save channel report in specified formats."""
        if formats is None:
            formats = ["json"]

        saved_files = []

        # Save JSON report
        if "json" in formats:
            json_file_path = Path(self.output_dir) / REPORT_FILE_PATTERN.format(
                channel=channel
            )
            pd.DataFrame([report]).to_json(
                json_file_path, orient="records", indent=2, default_handler=str
            )
            saved_files.append(str(json_file_path))

        # Save summary text file
        if "txt" in formats or "summary" in formats:
            summary_text = self._generate_summary_text(report, channel)
            summary_file_path = Path(self.output_dir) / SUMMARY_FILE_PATTERN.format(
                channel=channel
            )
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            saved_files.append(str(summary_file_path))

        # Save CSV export
        if "csv" in formats:
            csv_file_path = Path(self.output_dir) / f"{channel}_data.csv"
            df.to_csv(csv_file_path, index=False, encoding="utf-8")
            saved_files.append(str(csv_file_path))

        # Save Excel export
        if "excel" in formats or "xlsx" in formats:
            excel_file_path = Path(self.output_dir) / f"{channel}_data.xlsx"
            df.to_excel(excel_file_path, index=False, engine="openpyxl")
            saved_files.append(str(excel_file_path))

        return saved_files

    def _generate_summary_text(self, report: dict[str, Any], channel: str) -> str:
        """Generate a human-readable summary text file."""
        lines = []

        # Header
        lines.append(f"Channel Report Summary: {channel}")
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
        lines.append(f"Average File Size: {report.get('average_file_size', 'N/A')}")
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
