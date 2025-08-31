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
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from modules.database_service import TelegramDBService


class ChannelReporter:
    """Generates comprehensive reports for individual Telegram channels using pandas."""
    
    def __init__(self, db_service: TelegramDBService, output_dir: str = "./reports/channels"):
        self.db_service = db_service
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async def generate_channel_report(self, channel: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive report for a specific channel."""
        try:
            self.logger.info(f"📊 Generating comprehensive report for channel: {channel}")
            
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
                
                # Save report to files
                saved_files = await self._save_channel_report_pandas(channel, report, df)
                
                report['saved_files'] = saved_files
                return report
                
        except Exception as e:
            self.logger.error(f"Error generating report for channel {channel}: {e}")
            return None
    
    async def _fetch_channel_messages(self, channel: str) -> List[Dict[str, Any]]:
        """Fetch all messages for a specific channel."""
        try:
            # Use the bulk export endpoint for efficiency
            url = f"{self.db_service.db_url}/api/v1/telegram/messages/export/all"
            params = {
                'fields': 'message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at',
                'channel': channel
            }
            
            if self.db_service.session:
                async with self.db_service.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Filter for the specific channel
                        channel_messages = [msg for msg in result if msg.get('channel_username') == channel]
                        return channel_messages
                    else:
                        self.logger.warning(f"Bulk export failed for {channel}, falling back to paginated approach")
                        return await self._fetch_channel_messages_paginated(channel)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching messages for {channel}: {e}")
            return []
    
    async def _fetch_channel_messages_paginated(self, channel: str) -> List[Dict[str, Any]]:
        """Fallback: Fetch channel messages using paginated approach."""
        try:
            messages = []
            offset = 0
            limit = 1000
            
            while True:
                url = f"{self.db_service.db_url}/api/v1/telegram/messages"
                params = {
                    'limit': limit,
                    'offset': offset,
                    'channel': channel,
                    'fields': 'message_id,channel_username,date,text,media_type,file_name,file_size,mime_type,caption,views,forwards,replies,created_at,updated_at'
                }
                
                if self.db_service.session:
                    async with self.db_service.session.get(url, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch = result.get('messages', []) if 'messages' in result else result
                            
                            if not batch:
                                break
                            
                            messages.extend(batch)
                            offset += len(batch)
                            
                            if len(batch) < limit:
                                break
                        else:
                            break
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error in paginated fetch for {channel}: {e}")
            return []
    
    def _create_channel_report_pandas(self, channel: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive channel report using pandas operations."""
        try:
            self.logger.info(f"🔍 Creating comprehensive pandas report for {channel}")
            
            # Basic channel info
            report = {
                'channel_info': self._get_channel_info_pandas(channel, df),
                'message_statistics': self._get_message_statistics_pandas(df),
                'temporal_analysis': self._get_temporal_analysis_pandas(df),
                'content_analysis': self._get_content_analysis_pandas(df),
                'media_analysis': self._get_media_analysis_pandas(df),
                'engagement_analysis': self._get_engagement_analysis_pandas(df),
                'file_analysis': self._get_file_analysis_pandas(df),
                'correlation_analysis': self._get_correlation_analysis_pandas(df),
                'data_quality': self._get_data_quality_analysis_pandas(df),
                'generated_at': datetime.now().isoformat(),
                'data_summary': {
                    'total_messages': len(df),
                    'date_range': {
                        'start': df['date'].min() if 'date' in df.columns else None,
                        'end': df['date'].max() if 'date' in df.columns else None
                    },
                    'columns_available': list(df.columns)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating pandas report for {channel}: {e}")
            return {}
    
    def _get_channel_info_pandas(self, channel: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic channel information using pandas."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = df['date'].agg(['min', 'max'])
            active_days = (date_range['max'] - date_range['min']).days
        else:
            date_range = {'min': None, 'max': None}
            active_days = 0
        
        return {
            'channel_name': channel,
            'total_messages': len(df),
            'first_message_date': date_range['min'].isoformat() if date_range['min'] else None,
            'last_message_date': date_range['max'].isoformat() if date_range['max'] else None,
            'active_days': active_days
        }
    
    def _get_message_statistics_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive message statistics using pandas."""
        # Use pandas value_counts and boolean operations
        media_count = df['media_type'].notna().sum() if 'media_type' in df.columns else 0
        text_count = df['text'].notna().sum() if 'text' in df.columns else 0
        empty_count = df['text'].isna().sum() if 'text' in df.columns else 0
        total = len(df)
        
        stats = {
            'total_messages': total,
            'media_messages': media_count,
            'text_only_messages': text_count,
            'empty_messages': empty_count
        }
        
        # Calculate percentages using pandas
        if total > 0:
            stats['media_percentage'] = round((media_count / total) * 100, 2)
            stats['text_percentage'] = round((text_count / total) * 100, 2)
            stats['empty_percentage'] = round((empty_count / total) * 100, 2)
        
        return stats
    
    def _get_temporal_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns using pandas operations."""
        if 'date' not in df.columns:
            return {'error': 'Date column not available'}
        
        try:
            # Convert to datetime using pandas and remove timezone info for Excel compatibility
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Create pandas Series for time-based analysis
            monthly_series = df.groupby(df['date'].dt.to_period('M')).size()
            daily_series = df.groupby(df['date'].dt.date).size()
            hourly_series = df.groupby(df['date'].dt.hour).size()
            
            # Convert to pandas-friendly format for serialization
            monthly_df = monthly_series.reset_index()
            monthly_df.columns = ['period', 'count']
            monthly_df['period'] = monthly_df['period'].astype(str)
            
            daily_df = daily_series.reset_index()
            daily_df.columns = ['date', 'count']
            daily_df['date'] = daily_df['date'].astype(str)
            
            hourly_df = hourly_series.reset_index()
            hourly_df.columns = ['hour', 'count']
            
            # Calculate statistics using pandas
            activity_stats = {
                'avg_daily_messages': round(daily_series.mean(), 2),
                'std_daily_messages': round(daily_series.std(), 2),
                'max_daily_messages': int(daily_series.max()),
                'min_daily_messages': int(daily_series.min()),
                'total_active_days': len(daily_series[daily_series > 0])
            }
            
            # Peak activity using pandas idxmax
            peak_month = monthly_series.idxmax() if not monthly_series.empty else None
            peak_day = daily_series.idxmax() if not daily_series.empty else None
            peak_hour = hourly_series.idxmax() if not hourly_series.empty else None
            
            # Activity trends using pandas
            trends = self._calculate_activity_trends_pandas(monthly_series)
            
            return {
                'monthly_activity': monthly_df.to_dict('records'),
                'daily_activity': daily_df.to_dict('records'),
                'hourly_activity': hourly_df.to_dict('records'),
                'peak_activity': {
                    'month': str(peak_month) if peak_month else None,
                    'day': str(peak_day) if peak_day else None,
                    'hour': int(peak_hour) if peak_hour is not None else None
                },
                'activity_stats': activity_stats,
                'activity_trends': trends
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_activity_trends_pandas(self, monthly_series: pd.Series) -> Dict[str, Any]:
        """Calculate activity trends using pandas operations."""
        if len(monthly_series) < 2:
            return {'trend': 'insufficient_data'}
        
        try:
            # Use pandas for trend calculation
            x = np.arange(len(monthly_series))
            y = monthly_series.values
            
            # Simple linear trend using numpy (pandas compatible)
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                trend = 'increasing'
            elif slope < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Calculate volatility using pandas
            volatility = round(monthly_series.std() / monthly_series.mean(), 3) if monthly_series.mean() > 0 else 0
            
            return {
                'trend': trend,
                'slope': round(slope, 3),
                'volatility': volatility,
                'trend_strength': 'strong' if abs(slope) > monthly_series.std() else 'weak'
            }
            
        except Exception as e:
            return {'trend': 'error', 'error': str(e)}
    
    def _get_content_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text content patterns using pandas."""
        if 'text' not in df.columns:
            return {'error': 'Text column not available'}
        
        try:
            # Use pandas string methods
            text_lengths = df['text'].str.len()
            
            # Create pandas Series for statistics
            text_stats = {
                'mean_length': round(text_lengths.mean(), 2),
                'median_length': round(text_lengths.median(), 2),
                'max_length': int(text_lengths.max()),
                'min_length': int(text_lengths.min()),
                'std_length': round(text_lengths.std(), 2)
            }
            
            # Word count analysis using pandas
            non_empty_df = df[df['text'].notna()]
            if not non_empty_df.empty:
                word_counts = non_empty_df['text'].str.split().str.len()
                text_stats.update({
                    'mean_words': round(word_counts.mean(), 2),
                    'median_words': round(word_counts.median(), 2),
                    'max_words': int(word_counts.max()),
                    'min_words': int(word_counts.min()),
                    'total_words': int(word_counts.sum())
                })
            
            # Content categories using pandas boolean indexing
            content_categories = {
                'very_short': len(df[text_lengths <= 10]),
                'short': len(df[(text_lengths > 10) & (text_lengths <= 50)]),
                'medium': len(df[(text_lengths > 50) & (text_lengths <= 200)]),
                'long': len(df[(text_lengths > 200) & (text_lengths <= 500)]),
                'very_long': len(df[text_lengths > 500])
            }
            
            # Text composition using pandas
            text_composition = {
                'empty_texts': df['text'].isna().sum(),
                'non_empty_texts': df['text'].notna().sum()
            }
            
            return {
                'text_statistics': text_stats,
                'content_categories': content_categories,
                'text_composition': text_composition
            }
            
        except Exception as e:
            self.logger.error(f"Error in content analysis: {e}")
            return {'error': str(e)}
    
    def _get_media_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze media-related patterns using pandas."""
        if 'media_type' not in df.columns:
            return {'error': 'Media type column not available'}
        
        try:
            # Use pandas value_counts
            media_distribution = df['media_type'].value_counts()
            media_percentage = (media_distribution / len(df) * 100).round(2)
            
            # File size analysis using pandas
            file_size_stats = {}
            if 'file_size' in df.columns:
                file_sizes = pd.to_numeric(df['file_size'], errors='coerce')
                if not file_sizes.isna().all():
                    file_size_stats = {
                        'mean_mb': round(file_sizes.mean() / (1024 * 1024), 2),
                        'median_mb': round(file_sizes.median() / (1024 * 1024), 2),
                        'std_mb': round(file_sizes.std() / (1024 * 1024), 2),
                        'min_mb': round(file_sizes.min() / (1024 * 1024), 2),
                        'max_mb': round(file_sizes.max() / (1024 * 1024), 2),
                        'total_size_gb': round(file_sizes.sum() / (1024 * 1024 * 1024), 2)
                    }
            
            # MIME type analysis using pandas
            mime_analysis = {}
            if 'mime_type' in df.columns:
                mime_distribution = df['mime_type'].value_counts()
                mime_analysis = {
                    'top_mime_types': mime_distribution.head(10).to_dict(),
                    'unique_mime_types': len(mime_distribution),
                    'most_common_mime': mime_distribution.index[0] if not mime_distribution.empty else None
                }
            
            # Media efficiency using pandas
            if 'date' in df.columns:
                # Ensure date is already converted and timezone-naive
                date_range = df['date'].agg(['min', 'max'])
                active_days = (date_range['max'] - date_range['min']).days
                avg_files_per_day = round(len(df) / max(1, active_days), 2)
            else:
                avg_files_per_day = 0
            
            return {
                'media_distribution': media_distribution.to_dict(),
                'media_percentage': media_percentage.to_dict(),
                'file_size_statistics': file_size_stats,
                'mime_type_analysis': mime_analysis,
                'media_efficiency': {
                    'avg_files_per_day': avg_files_per_day
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in media analysis: {e}")
            return {'error': str(e)}
    
    def _get_engagement_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement metrics using pandas."""
        engagement_metrics = {}
        
        try:
            # Views analysis using pandas
            if 'views' in df.columns:
                views = pd.to_numeric(df['views'], errors='coerce')
                if not views.isna().all():
                    engagement_metrics['views'] = {
                        'mean': round(views.mean(), 2),
                        'median': round(views.median(), 2),
                        'max': int(views.max()),
                        'min': int(views.min()),
                        'total': int(views.sum()),
                        'engagement_rate': round((views > 0).mean() * 100, 2)
                    }
            
            # Forwards analysis using pandas
            if 'forwards' in df.columns:
                forwards = pd.to_numeric(df['forwards'], errors='coerce')
                if not forwards.isna().all():
                    engagement_metrics['forwards'] = {
                        'mean': round(forwards.mean(), 2),
                        'median': round(forwards.median(), 2),
                        'max': int(forwards.max()),
                        'min': int(forwards.min()),
                        'total': int(forwards.sum()),
                        'forward_rate': round((forwards > 0).mean() * 100, 2)
                    }
            
            # Replies analysis using pandas
            if 'replies' in df.columns:
                replies = pd.to_numeric(df['replies'], errors='coerce')
                if not replies.isna().all():
                    engagement_metrics['replies'] = {
                        'mean': round(replies.mean(), 2),
                        'median': round(replies.median(), 2),
                        'max': int(replies.max()),
                        'min': int(replies.min()),
                        'total': int(replies.sum()),
                        'reply_rate': round((replies > 0).mean() * 100, 2)
                    }
            
            # Overall engagement using pandas
            if engagement_metrics:
                total_engagement = sum(metric.get('total', 0) for metric in engagement_metrics.values())
                engagement_metrics['overall'] = {
                    'total_engagement': total_engagement,
                    'avg_engagement_per_message': round(total_engagement / len(df), 2) if len(df) > 0 else 0
                }
            
            return engagement_metrics
            
        except Exception as e:
            self.logger.error(f"Error in engagement analysis: {e}")
            return {}
    
    def _get_file_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze file-related patterns using pandas."""
        file_patterns = {}
        
        try:
            # File name analysis using pandas
            if 'file_name' in df.columns:
                file_names = df['file_name'].dropna()
                if not file_names.empty:
                    # File extensions using pandas string methods
                    extensions = file_names.str.extract(r'\.([^.]+)$')[0].value_counts()
                    file_patterns['extensions'] = extensions.head(15).to_dict()
                    
                    # File name lengths using pandas
                    name_lengths = file_names.str.len()
                    file_patterns['name_lengths'] = {
                        'mean': round(name_lengths.mean(), 2),
                        'median': round(name_lengths.median(), 2),
                        'max': int(name_lengths.max()),
                        'min': int(name_lengths.min())
                    }
                    
                    # File naming patterns using pandas string methods
                    file_patterns['naming_patterns'] = {
                        'has_numbers': (file_names.str.contains(r'\d')).sum(),
                        'has_underscores': (file_names.str.contains(r'_')).sum(),
                        'has_hyphens': (file_names.str.contains(r'-')).sum(),
                        'has_spaces': (file_names.str.contains(r'\s')).sum()
                    }
            
            # Caption analysis using pandas
            if 'caption' in df.columns:
                captions = df['caption'].dropna()
                if not captions.empty:
                    caption_lengths = captions.str.len()
                    file_patterns['captions'] = {
                        'count': len(captions),
                        'mean_length': round(caption_lengths.mean(), 2),
                        'median_length': round(caption_lengths.median(), 2),
                        'max_length': int(caption_lengths.max()),
                        'min_length': int(caption_lengths.min()),
                        'caption_rate': round(len(captions) / len(df) * 100, 2)
                    }
            
            return file_patterns
            
        except Exception as e:
            self.logger.error(f"Error in file analysis: {e}")
            return {}
    
    def _get_correlation_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations using pandas operations."""
        try:
            correlations = {}
            
            # Select numeric columns using pandas
            numeric_cols = df.select_dtypes(include=[pd.Int64Dtype(), pd.Float64Dtype()]).columns
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix using pandas
                corr_matrix = df[numeric_cols].corr()
                
                # Find strong correlations using pandas operations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5 and not pd.isna(corr_value):
                            strong_correlations.append({
                                'variable1': corr_matrix.columns[i],
                                'variable2': corr_matrix.columns[j],
                                'correlation': round(corr_value, 3),
                                'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                            })
                
                correlations['strong_correlations'] = strong_correlations
                correlations['correlation_matrix'] = corr_matrix.round(3).to_dict()
                correlations['correlation_summary'] = {
                    'total_correlations': len(strong_correlations),
                    'strong_correlations': len([c for c in strong_correlations if c['strength'] == 'strong']),
                    'moderate_correlations': len([c for c in strong_correlations if c['strength'] == 'moderate'])
                }
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _get_data_quality_analysis_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality using pandas operations."""
        try:
            # Missing values analysis using pandas
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df) * 100)
            
            # Duplicate analysis using pandas
            duplicates = df.duplicated().sum()
            
            # Data type analysis using pandas
            data_types = df.dtypes.to_dict()
            
            # Column completeness using pandas
            column_completeness = {}
            for col in df.columns:
                non_null_count = df[col].notna().sum()
                column_completeness[col] = {
                    'non_null_count': int(non_null_count),
                    'null_count': int(len(df) - non_null_count),
                    'completeness_percentage': round((non_null_count / len(df)) * 100, 2)
                }
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score_pandas(df)
            
            return {
                'missing_values': missing_data.to_dict(),
                'missing_percentage': missing_percentage.to_dict(),
                'duplicate_rows': int(duplicates),
                'duplicate_percentage': round((duplicates / len(df)) * 100, 2),
                'completeness_score': round((1 - (missing_data.sum() / (len(df) * len(df.columns)))) * 100, 2),
                'data_types': {col: str(dtype) for col, dtype in data_types.items()},
                'column_completeness': column_completeness,
                'data_quality_score': data_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in data quality analysis: {e}")
            return {}
    
    def _calculate_data_quality_score_pandas(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score using pandas."""
        try:
            # Completeness score (0-40 points) using pandas
            completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            completeness_score = completeness * 40
            
            # Consistency score (0-30 points)
            consistency_score = 30  # Simplified for now
            
            # Accuracy score (0-30 points)
            accuracy_score = 30  # Simplified for now
            
            total_score = completeness_score + consistency_score + accuracy_score
            return round(total_score, 2)
            
        except Exception:
            return 0.0
    
    async def _save_channel_report_pandas(self, channel: str, report: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
        """Save channel report to various file formats using pandas."""
        try:
            safe_channel_name = channel.replace('@', '').replace('/', '_').replace(' ', '_')
            channel_dir = os.path.join(self.output_dir, safe_channel_name)
            Path(channel_dir).mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save JSON report (pandas-compatible)
            json_file = os.path.join(channel_dir, f"{safe_channel_name}_report.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            saved_files['json'] = json_file
            
            # Save CSV data using pandas
            csv_file = os.path.join(channel_dir, f"{safe_channel_name}_data.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8')
            saved_files['csv'] = csv_file
            
            # Save Excel file with multiple sheets using pandas
            excel_file = os.path.join(channel_dir, f"{safe_channel_name}_report.xlsx")
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # Summary statistics
                summary_data = []
                for category, data in report.items():
                    if isinstance(data, dict) and category != 'saved_files':
                        summary_data.append([category, json.dumps(data, default=str)])
                
                summary_df = pd.DataFrame(summary_data, columns=['Category', 'Data'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Temporal analysis sheets using pandas DataFrames
                if 'temporal_analysis' in report and 'error' not in report['temporal_analysis']:
                    temp_data = report['temporal_analysis']
                    
                    if 'monthly_activity' in temp_data:
                        monthly_df = pd.DataFrame(temp_data['monthly_activity'])
                        monthly_df.to_excel(writer, sheet_name='Monthly_Activity', index=False)
                    
                    if 'daily_activity' in temp_data:
                        daily_df = pd.DataFrame(temp_data['daily_activity'])
                        daily_df.to_excel(writer, sheet_name='Daily_Activity', index=False)
                    
                    if 'hourly_activity' in temp_data:
                        hourly_df = pd.DataFrame(temp_data['hourly_activity'])
                        hourly_df.to_excel(writer, sheet_name='Hourly_Activity', index=False)
                
                # Media analysis sheet
                if 'media_analysis' in report and 'error' not in report['media_analysis']:
                    media_data = report['media_analysis']
                    if 'media_distribution' in media_data:
                        media_df = pd.DataFrame(list(media_data['media_distribution'].items()), 
                                              columns=['Media_Type', 'Count'])
                        media_df.to_excel(writer, sheet_name='Media_Distribution', index=False)
            
            saved_files['excel'] = excel_file
            
            # Save summary text file
            summary_file = os.path.join(channel_dir, f"{safe_channel_name}_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Channel Report Summary: {channel}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Messages: {len(df):,}\n")
                f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
                f.write(f"Media Messages: {df['media_type'].notna().sum():,}\n")
                f.write(f"Text Messages: {df['text'].notna().sum():,}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            saved_files['summary'] = summary_file
            
            self.logger.info(f"✅ Channel report saved to: {channel_dir}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving channel report for {channel}: {e}")
            return {}
    
    async def generate_all_channel_reports(self, channels: List[str]) -> Dict[str, Any]:
        """Generate reports for all specified channels."""
        results = {}
        
        for channel in channels:
            try:
                self.logger.info(f"🔄 Processing channel: {channel}")
                report = await self.generate_channel_report(channel)
                if report:
                    results[channel] = {
                        'status': 'success',
                        'report': report
                    }
                else:
                    results[channel] = {
                        'status': 'failed',
                        'error': 'No data available'
                    }
            except Exception as e:
                results[channel] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
