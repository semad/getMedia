"""
Analyze Processor for Telegram message analysis.

This module now uses a separated architecture:
- StatsService: Data collection and statistics calculation
- DashboardRenderer: HTML generation and template rendering
- FileManager: File operations and I/O management
- DashboardOrchestrator: Coordinates all services
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from modules.dashboard_orchestrator import DashboardOrchestrator


class AnalyzeProcessor:
    """Main processor for analyzing Telegram messages and generating dashboards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.orchestrator = None
    
    async def analyze_database(self, generate_dashboard: bool = True) -> Optional[Dict[str, Any]]:
        """Analyze database and optionally generate dashboard."""
        try:
            self.logger.info("🔍 DATABASE ANALYSIS MODE - Analyzing database statistics")
            
            # Initialize orchestrator
            self.orchestrator = DashboardOrchestrator()
            
            # Get database statistics
            self.logger.info("Fetching database statistics...")
            stats = await self.orchestrator.stats_service.get_database_stats()
            
            if not stats:
                self.logger.error("Failed to fetch database statistics")
                return None
            
            # Log statistics
            self._log_database_stats(stats)
            
            # Generate dashboard if requested
            if generate_dashboard:
                self.logger.info("Generating comprehensive dashboard system...")
                summary = await self.orchestrator.generate_complete_dashboard_system()
                
                self.logger.info(f"✅ Complete dashboard system generated in: {summary['output_directory']}")
                self.logger.info(f"📊 HTML dashboard saved to {summary['output_directory']} directory")
                
                return summary
            
            return {'stats': stats}
            
        except Exception as e:
            self.logger.error(f"Database analysis failed: {e}")
            return None
    
    def _log_database_stats(self, stats: Dict[str, Any]):
        """Log database statistics in a formatted way."""
        self.logger.info("=== DATABASE STATISTICS ===")
        self.logger.info(f"📊 Total Messages: {stats.get('total_messages', 0):,}")
        self.logger.info(f"📺 Total Channels: {stats.get('total_channels', 0):,}")
        self.logger.info(f"💾 Total Storage: {stats.get('total_storage_gb', 0.0):.2f} GB")
        
        # Media breakdown
        media_breakdown = stats.get('media_breakdown', {})
        if media_breakdown:
            self.logger.info("📁 Media Breakdown:")
            for media_type, count in media_breakdown.items():
                self.logger.info(f"   {media_type}: {count:,}")
        
        # Channel breakdown
        channel_breakdown = stats.get('channel_breakdown', {})
        if channel_breakdown:
            self.logger.info("📺 Channel Breakdown:")
            for channel, count in channel_breakdown.items():
                self.logger.info(f"   {channel}: {count:,}")
        
        # Date range
        date_range = stats.get('date_range', {})
        if date_range:
            start_date = date_range.get('start', 'N/A')
            end_date = date_range.get('end', 'N/A')
            self.logger.info(f"📅 Date Range: {start_date} to {end_date}")
        
        # Pandas Analysis Results
        self._log_pandas_analysis(stats)
    
    def _log_pandas_analysis(self, stats: Dict[str, Any]):
        """Log the comprehensive pandas analysis results."""
        self.logger.info("=== PANDAS ANALYSIS RESULTS ===")
        
        # DataFrame Info
        df_info = stats.get('dataframe_info', {})
        if df_info:
            self.logger.info("📊 DataFrame Information:")
            self.logger.info(f"   Total Rows: {df_info.get('total_rows', 0):,}")
            self.logger.info(f"   Total Columns: {df_info.get('total_columns', 0)}")
            self.logger.info(f"   Memory Usage: {df_info.get('memory_usage_mb', 0)} MB")
            self.logger.info(f"   Columns: {', '.join(df_info.get('columns', []))}")
        
        # Data Quality
        data_quality = stats.get('data_quality', {})
        if data_quality:
            self.logger.info("🔍 Data Quality Analysis:")
            self.logger.info(f"   Completeness Score: {data_quality.get('completeness_score', 0)}%")
            self.logger.info(f"   Duplicate Rows: {data_quality.get('duplicate_rows', 0):,} ({data_quality.get('duplicate_percentage', 0)}%)")
            
            missing_data = data_quality.get('missing_values', {})
            if missing_data:
                self.logger.info("   Missing Values by Column:")
                for col, count in missing_data.items():
                    if count > 0:
                        percentage = data_quality.get('missing_percentage', {}).get(col, 0)
                        self.logger.info(f"     {col}: {count:,} ({percentage:.1f}%)")
        
        # Temporal Analysis
        temporal = stats.get('temporal_analysis', {})
        if temporal and 'error' not in temporal:
            self.logger.info("⏰ Temporal Patterns:")
            date_range = temporal.get('date_range', {})
            if date_range:
                self.logger.info(f"   Total Days: {date_range.get('total_days', 0):,}")
            
            activity_stats = temporal.get('activity_stats', {})
            if activity_stats:
                self.logger.info(f"   Average Daily Messages: {activity_stats.get('avg_daily_messages', 0):.1f}")
                self.logger.info(f"   Max Daily Messages: {activity_stats.get('max_daily_messages', 0):,}")
            
            peak_activity = temporal.get('peak_activity', {})
            if peak_activity:
                self.logger.info(f"   Peak Activity - Month: {peak_activity.get('month', 'N/A')}")
                self.logger.info(f"   Peak Activity - Day: {peak_activity.get('day', 'N/A')}")
                self.logger.info(f"   Peak Activity - Hour: {peak_activity.get('hour', 'N/A')}")
        
        # Channel Analysis
        channel_analysis = stats.get('channel_analysis', {})
        if channel_analysis and 'error' not in channel_analysis:
            self.logger.info("📺 Channel Analysis:")
            most_active = channel_analysis.get('most_active_channel')
            least_active = channel_analysis.get('least_active_channel')
            if most_active:
                self.logger.info(f"   Most Active Channel: {most_active}")
            if least_active:
                self.logger.info(f"   Least Active Channel: {least_active}")
        
        # Media Analysis
        media_analysis = stats.get('media_analysis', {})
        if media_analysis and 'error' not in media_analysis:
            self.logger.info("📁 Media Analysis:")
            media_ratio = media_analysis.get('media_ratio', 0)
            self.logger.info(f"   Media Messages Ratio: {media_ratio}%")
            
            file_stats = media_analysis.get('file_size_stats', {})
            if file_stats:
                self.logger.info(f"   File Size - Mean: {file_stats.get('mean_mb', 0)} MB")
                self.logger.info(f"   File Size - Median: {file_stats.get('median_mb', 0)} MB")
                self.logger.info(f"   File Size - Range: {file_stats.get('min_mb', 0)} - {file_stats.get('max_mb', 0)} MB")
        
        # Text Analysis
        text_analysis = stats.get('text_analysis', {})
        if text_analysis and 'error' not in text_analysis:
            self.logger.info("📝 Text Analysis:")
            text_stats = text_analysis.get('text_stats', {})
            if text_stats:
                self.logger.info(f"   Text Length - Mean: {text_stats.get('mean_length', 0):.1f} characters")
                self.logger.info(f"   Text Length - Median: {text_stats.get('median_length', 0):.1f} characters")
                self.logger.info(f"   Word Count - Mean: {text_stats.get('mean_words', 0):.1f} words")
            
            text_composition = text_analysis.get('text_composition', {})
            if text_composition:
                self.logger.info(f"   Text Messages: {text_composition.get('text_percentage', 0)}%")
        
        # Engagement Analysis
        engagement = stats.get('engagement_analysis', {})
        if engagement:
            self.logger.info("👥 Engagement Analysis:")
            for metric, data in engagement.items():
                if isinstance(data, dict) and 'total' in data:
                    self.logger.info(f"   {metric.title()} - Total: {data['total']:,}, Mean: {data['mean']:.1f}")
        
        # File Analysis
        file_analysis = stats.get('file_analysis', {})
        if file_analysis:
            self.logger.info("📂 File Analysis:")
            extensions = file_analysis.get('extensions', {})
            if extensions:
                top_extensions = dict(list(extensions.items())[:5])
                self.logger.info(f"   Top File Extensions: {', '.join([f'{ext} ({count:,})' for ext, count in top_extensions.items()])}")
        
        # Correlations
        correlations = stats.get('correlations', {})
        if correlations:
            strong_corr = correlations.get('strong_correlations', [])
            if strong_corr:
                self.logger.info("🔗 Strong Correlations (|r| > 0.5):")
                for corr in strong_corr[:5]:  # Show top 5
                    self.logger.info(f"   {corr['variable1']} ↔ {corr['variable2']}: r = {corr['correlation']}")
        
        self.logger.info("=" * 50)
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get current dashboard status."""
        if self.orchestrator:
            return self.orchestrator.get_dashboard_status()
        return {}
    
    def validate_templates(self) -> Dict[str, bool]:
        """Validate all templates."""
        if self.orchestrator:
            return self.orchestrator.validate_templates()
        return {}
    
    def test_rendering(self) -> Dict[str, bool]:
        """Test rendering with sample data."""
        if self.orchestrator:
            return self.orchestrator.test_rendering()
        return {}
    
    def cleanup_old_dashboards(self, max_age_hours: int = 24) -> int:
        """Clean up old dashboard files."""
        if self.orchestrator:
            return self.orchestrator.cleanup_old_dashboards(max_age_hours)
        return 0


# Legacy functions for backward compatibility
async def analyze_database(generate_dashboard: bool = True) -> Optional[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    processor = AnalyzeProcessor()
    return await processor.analyze_database(generate_dashboard)


def generate_channel_dashboard(channel: str, message_count: int, stats: Dict[str, Any]) -> str:
    """Legacy function for backward compatibility."""
    # This function is now handled by DashboardRenderer
    # Keeping for backward compatibility
    return f"<html><body><h1>{channel}</h1><p>Legacy function - use DashboardRenderer instead</p></body></html>"


def generate_index_dashboard(stats: Dict[str, Any]) -> str:
    """Legacy function for backward compatibility."""
    # This function is now handled by DashboardRenderer
    # Keeping for backward compatibility
    return f"<html><body><h1>Index Dashboard</h1><p>Legacy function - use DashboardRenderer instead</p></body></html>"
