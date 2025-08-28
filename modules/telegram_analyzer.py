"""
Telegram data analyzer for analyzing message data and generating reports.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TelegramDataAnalyzer:
    """Analyzes Telegram message data and generates reports."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self):
        """Load data from file."""
        try:
            file_path = Path(self.data_file)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    self.df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'messages' in data:
                    self.df = pd.DataFrame(data['messages'])
                else:
                    logger.error("Invalid JSON structure")
                    return
                    
            elif file_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return
            
            # Convert date columns
            date_columns = ['date', 'edit_date', 'created_at', 'updated_at']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            logger.info(f"Loaded {len(self.df)} messages from {self.data_file}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def generate_basic_stats(self) -> Dict[str, Any]:
        """Generate basic statistics."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}
        
        stats = {
            "total_messages": len(self.df),
            "unique_channels": self.df['channel_username'].nunique() if 'channel_username' in self.df.columns else 0,
            "date_range": {
                "start": self.df['date'].min().isoformat() if 'date' in self.df.columns else None,
                "end": self.df['date'].max().isoformat() if 'date' in self.df.columns else None
            }
        }
        
        # Channel breakdown
        if 'channel_username' in self.df.columns:
            channel_counts = self.df['channel_username'].value_counts()
            stats["top_channels"] = channel_counts.head(5).to_dict()
        
        # Media type breakdown
        if 'media_type' in self.df.columns:
            media_counts = self.df['media_type'].value_counts()
            stats["media_types"] = media_counts.to_dict()
        
        return stats
    
    def generate_channel_report(self) -> str:
        """Generate a channel analysis report."""
        if self.df is None or self.df.empty:
            return "No data to analyze"
        
        if 'channel_username' not in self.df.columns:
            return "No channel information available"
        
        report = []
        report.append("=" * 60)
        report.append("CHANNEL ANALYSIS REPORT")
        report.append("=" * 60)
        
        channel_stats = self.df.groupby('channel_username').agg({
            'message_id': 'count',
            'date': ['min', 'max']
        }).round(2)
        
        channel_stats.columns = ['Message Count', 'First Message', 'Last Message']
        channel_stats = channel_stats.sort_values('Message Count', ascending=False)
        
        report.append(f"Total Channels: {len(channel_stats)}")
        report.append(f"Total Messages: {channel_stats['Message Count'].sum()}")
        report.append("\nTop Channels by Message Count:")
        
        for channel, row in channel_stats.head(10).iterrows():
            report.append(f"  {channel}: {row['Message Count']} messages")
            report.append(f"    Date Range: {row['First Message']} to {row['Last Message']}")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def generate_media_report(self) -> str:
        """Generate a media analysis report."""
        if self.df is None or self.df.empty:
            return "No data to analyze"
        
        if 'media_type' not in self.df.columns:
            return "No media information available"
        
        report = []
        report.append("=" * 60)
        report.append("MEDIA ANALYSIS REPORT")
        report.append("=" * 60)
        
        media_stats = self.df['media_type'].value_counts()
        total_messages = len(self.df)
        
        report.append(f"Total Messages: {total_messages}")
        report.append(f"Messages with Media: {media_stats.sum()}")
        report.append(f"Text-only Messages: {total_messages - media_stats.sum()}")
        
        report.append("\nMedia Type Breakdown:")
        for media_type, count in media_stats.items():
            percentage = (count / total_messages) * 100
            report.append(f"  {media_type}: {count} messages ({percentage:.1f}%)")
        
        # File size analysis if available
        if 'file_size' in self.df.columns:
            media_messages = self.df[self.df['media_type'].notna()]
            if not media_messages.empty:
                file_sizes = media_messages['file_size'].dropna()
                if not file_sizes.empty:
                    report.append(f"\nFile Size Statistics:")
                    report.append(f"  Average: {file_sizes.mean():.0f} bytes")
                    report.append(f"  Median: {file_sizes.median():.0f} bytes")
                    report.append(f"  Largest: {file_sizes.max():.0f} bytes")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def generate_temporal_report(self) -> str:
        """Generate a temporal analysis report."""
        if self.df is None or self.df.empty:
            return "No data to analyze"
        
        if 'date' not in self.df.columns:
            return "No date information available"
        
        report = []
        report.append("=" * 60)
        report.append("TEMPORAL ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Date range
        date_range = self.df['date'].max() - self.df['date'].min()
        report.append(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        report.append(f"Total Duration: {date_range.days} days")
        
        # Monthly breakdown
        monthly_counts = self.df.groupby(self.df['date'].dt.to_period('M')).size()
        report.append(f"\nMonthly Message Count:")
        for month, count in monthly_counts.tail(12).items():
            report.append(f"  {month}: {count} messages")
        
        # Daily pattern
        daily_counts = self.df.groupby(self.df['date'].dt.day_name()).size()
        report.append(f"\nDaily Pattern (by day of week):")
        for day, count in daily_counts.items():
            report.append(f"  {day}: {count} messages")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if self.df is None or self.df.empty:
            return "No data to analyze"
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TELEGRAM MESSAGES ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Basic stats
        basic_stats = self.generate_basic_stats()
        report.append(f"Total Messages: {basic_stats.get('total_messages', 0)}")
        report.append(f"Unique Channels: {basic_stats.get('unique_channels', 0)}")
        
        # Channel report
        report.append("\n" + self.generate_channel_report())
        
        # Media report
        report.append("\n" + self.generate_media_report())
        
        # Temporal report
        report.append("\n" + self.generate_temporal_report())
        
        report.append("=" * 80)
        return "\n".join(report)
