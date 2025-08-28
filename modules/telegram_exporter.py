"""
Telegram message exporter for exporting data to various formats.
"""

import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import TelegramMessage

logger = logging.getLogger(__name__)


class TelegramMessageExporter:
    """Exports Telegram messages to various formats."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_total_count(self) -> int:
        """Get total count of messages in database."""
        # This would need to be implemented based on your database schema
        # For now, return a placeholder
        return 0
    
    async def get_all_messages(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Get all messages from database in batches."""
        # This would need to be implemented based on your database schema
        # For now, return an empty list
        return []
    
    def create_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert messages to pandas DataFrame."""
        if not messages:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(messages)
        
        # Convert date columns to datetime
        date_columns = ['date', 'edit_date', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a summary report of the data."""
        if df.empty:
            return "No data to analyze"
        
        report = []
        report.append("=" * 60)
        report.append("TELEGRAM MESSAGES SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Total Messages: {len(df)}")
        report.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        report.append(f"Unique Channels: {df['channel_username'].nunique()}")
        
        # Channel breakdown
        channel_counts = df['channel_username'].value_counts()
        report.append("\nChannel Breakdown:")
        for channel, count in channel_counts.head(10).items():
            report.append(f"  {channel}: {count} messages")
        
        # Media type breakdown
        if 'media_type' in df.columns:
            media_counts = df['media_type'].value_counts()
            report.append("\nMedia Type Breakdown:")
            for media_type, count in media_counts.head(10).items():
                report.append(f"  {media_type}: {count} messages")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def export_to_json(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to JSON format."""
        try:
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Exported {len(records)} messages to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to CSV format."""
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Exported {len(df)} messages to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def export_to_excel(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to Excel format."""
        try:
            df.to_excel(filename, index=False, engine='openpyxl')
            logger.info(f"Exported {len(df)} messages to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            return False
