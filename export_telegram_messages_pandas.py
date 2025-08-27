#!/usr/bin/env python3
"""
Telegram Messages Database Export Script with Pandas

This script exports all messages from the telegram_messages table
into various formats (JSON, CSV, Excel) using pandas for enhanced
data handling and analysis capabilities.

Usage:
    python export_telegram_messages_pandas.py [--output filename] [--format json|csv|excel|all] [--channel username]
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
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramMessageExporter:
    """Export Telegram messages from database using pandas."""
    
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
    
    def create_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert messages to pandas DataFrame with data cleaning and optimization."""
        if not messages:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(messages)
        
        # Data cleaning and optimization
        logger.info("Processing and cleaning data...")
        
        # Convert date columns to datetime
        date_columns = ['date', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle numeric columns
        numeric_columns = ['message_id', 'views', 'forwards', 'replies', 'file_size']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['text', 'caption', 'file_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Add derived columns for analysis
        df['text_length'] = df['text'].str.len()
        df['has_media'] = df['media_type'].notna() & (df['media_type'] != '')
        df['is_forwarded'] = df['is_forwarded'].fillna(False)
        
        # Reorder columns for better readability
        priority_columns = ['message_id', 'channel_username', 'date', 'text', 'media_type', 'file_name']
        other_columns = [col for col in df.columns if col not in priority_columns]
        df = df[priority_columns + other_columns]
        
        logger.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def export_to_json(self, df: pd.DataFrame, filename: str, format_type: str = "pretty") -> bool:
        """Export DataFrame to JSON format."""
        try:
            export_data = {
                "export_info": {
                    "exported_at": datetime.now(UTC).isoformat(),
                    "total_messages": len(df),
                    "format": "pandas_dataframe",
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict()
                },
                "messages": df.to_dict('records')
            }
            
            if format_type == "pretty":
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, default=str)
            
            logger.info(f"Data exported to JSON: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON {filename}: {e}")
            return False
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to CSV format."""
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Data exported to CSV: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV {filename}: {e}")
            return False
    
    def export_to_excel(self, df: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to Excel format with multiple sheets."""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Messages', index=False)
                
                # Summary statistics sheet
                summary_data = {
                    'Metric': [
                        'Total Messages',
                        'Unique Channels',
                        'Messages with Media',
                        'Forwarded Messages',
                        'Date Range Start',
                        'Date Range End',
                        'Average Text Length',
                        'Most Common Media Type'
                    ],
                    'Value': [
                        len(df),
                        df['channel_username'].nunique(),
                        df['has_media'].sum(),
                        df['is_forwarded'].sum(),
                        df['date'].min() if 'date' in df.columns else 'N/A',
                        df['date'].max() if 'date' in df.columns else 'N/A',
                        df['text_length'].mean() if 'text_length' in df.columns else 'N/A',
                        df['media_type'].mode().iloc[0] if 'media_type' in df.columns and not df['media_type'].empty else 'N/A'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Channel statistics sheet
                if 'channel_username' in df.columns:
                    channel_stats = df.groupby('channel_username').agg({
                        'message_id': 'count',
                        'has_media': 'sum',
                        'is_forwarded': 'sum',
                        'text_length': 'mean'
                    }).rename(columns={
                        'message_id': 'message_count',
                        'has_media': 'media_count',
                        'is_forwarded': 'forwarded_count',
                        'text_length': 'avg_text_length'
                    })
                    channel_stats.to_excel(writer, sheet_name='Channel Statistics')
                
                # Media type analysis sheet
                if 'media_type' in df.columns:
                    media_stats = df['media_type'].value_counts().reset_index()
                    media_stats.columns = ['Media Type', 'Count']
                    media_stats.to_excel(writer, sheet_name='Media Analysis', index=False)
            
            logger.info(f"Data exported to Excel: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Excel {filename}: {e}")
            return False
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive summary report of the data."""
        if df.empty:
            return "No data to analyze"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TELEGRAM MESSAGES DATA SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Basic statistics
        report_lines.append("BASIC STATISTICS:")
        report_lines.append(f"  Total Messages: {len(df):,}")
        report_lines.append(f"  Unique Channels: {df['channel_username'].nunique()}")
        report_lines.append(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
        report_lines.append("")
        
        # Channel breakdown
        report_lines.append("CHANNEL BREAKDOWN:")
        channel_counts = df['channel_username'].value_counts()
        for channel, count in channel_counts.head(10).items():
            report_lines.append(f"  {channel}: {count:,} messages")
        if len(channel_counts) > 10:
            report_lines.append(f"  ... and {len(channel_counts) - 10} more channels")
        report_lines.append("")
        
        # Media analysis
        if 'media_type' in df.columns:
            report_lines.append("MEDIA ANALYSIS:")
            media_counts = df['media_type'].value_counts()
            for media_type, count in media_counts.items():
                if pd.notna(media_type) and media_type:
                    report_lines.append(f"  {media_type}: {count:,} messages")
            report_lines.append(f"  No Media: {df['media_type'].isna().sum():,} messages")
            report_lines.append("")
        
        # Text analysis
        if 'text_length' in df.columns:
            report_lines.append("TEXT ANALYSIS:")
            report_lines.append(f"  Average Text Length: {df['text_length'].mean():.1f} characters")
            report_lines.append(f"  Shortest Message: {df['text_length'].min()} characters")
            report_lines.append(f"  Longest Message: {df['text_length'].max()} characters")
            report_lines.append("")
        
        # Forwarding analysis
        if 'is_forwarded' in df.columns:
            report_lines.append("FORWARDING ANALYSIS:")
            forwarded_count = df['is_forwarded'].sum()
            report_lines.append(f"  Forwarded Messages: {forwarded_count:,} ({forwarded_count/len(df)*100:.1f}%)")
            report_lines.append(f"  Original Messages: {len(df) - forwarded_count:,} ({(len(df) - forwarded_count)/len(df)*100:.1f}%)")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


@click.command(help='Export Telegram messages from database using pandas')
@click.option('--output', '-o', default='telegram_messages_export',
              help='Output filename without extension (default: telegram_messages_export)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'excel', 'all']), default='json',
              help='Export format: json, csv, excel, or all (default: json)')
@click.option('--channel', '-c', help='Export only messages from specific channel username')
@click.option('--batch-size', '-b', default=1000, type=int,
              help='Batch size for fetching messages (default: 1000)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--summary', '-s', is_flag=True, help='Generate and display summary report')
def main(output: str, format: str, channel: str, batch_size: int, verbose: bool, summary: bool):
    """Export all Telegram messages from database using pandas."""
    
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
                
                # Convert to pandas DataFrame
                df = exporter.create_dataframe(messages)
                
                # Generate summary report if requested
                if summary:
                    report = exporter.generate_summary_report(df)
                    print("\n" + report)
                
                # Export based on format
                success = False
                if format == 'json' or format == 'all':
                    json_filename = f"{output}.json"
                    success = exporter.export_to_json(df, json_filename)
                
                if format == 'csv' or format == 'all':
                    csv_filename = f"{output}.csv"
                    success = exporter.export_to_csv(df, csv_filename)
                
                if format == 'excel' or format == 'all':
                    excel_filename = f"{output}.xlsx"
                    success = exporter.export_to_excel(df, excel_filename)
                
                if success:
                    logger.info(f"Export completed successfully!")
                    logger.info(f"Output file(s): {output}.*")
                    logger.info(f"Total messages exported: {len(messages)}")
                    
                    # Show DataFrame info
                    logger.info(f"DataFrame shape: {df.shape}")
                    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                else:
                    logger.error("Export failed!")
                    
            except Exception as e:
                logger.error(f"Export failed with error: {e}")
                sys.exit(1)
    
    # Run the export
    asyncio.run(run_export())


if __name__ == '__main__':
    main()
