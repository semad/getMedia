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
        report.append("COMPREHENSIVE TELEGRAM ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic stats
        basic_stats = self.generate_basic_stats()
        report.append(f"Total Messages: {basic_stats.get('total_messages', 0)}")
        report.append(f"Unique Channels: {basic_stats.get('unique_channels', 0)}")
        if 'date_range' in basic_stats and basic_stats['date_range']['start']:
            report.append(f"Date Range: {basic_stats['date_range']['start']} to {basic_stats['date_range']['end']}")
        report.append("")
        
        # Add channel report
        report.append(self.generate_channel_report())
        report.append("")
        
        # Add media report
        report.append(self.generate_media_report())
        report.append("")
        
        # Add temporal report
        report.append(self.generate_temporal_report())
        
        return "\n".join(report)
    
    def generate_interactive_dashboard(self, channel_name: Optional[str] = None) -> str:
        """Generate an interactive HTML dashboard using Plotly."""
        if self.df is None or self.df.empty:
            return "<html><body><h1>No data available for dashboard</h1></body></html>"
        
        # Set up the dashboard title
        dashboard_title = f"Telegram Analysis Dashboard - {channel_name}" if channel_name else "Telegram Analysis Dashboard"
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            from config import DEFAULT_GA_MEASUREMENT_ID
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Message Activity Over Time', 'Media Type Distribution', 
                              'File Size Distribution', 'Hourly Activity Pattern',
                              'Daily Activity Pattern', 'Channel Overview'),
                specs=[[{"type": "scatter"}, {"type": "pie"}],
                       [{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # 1. Message Activity Over Time
            if 'date' in self.df.columns:
                date_counts = self.df['date'].value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(x=date_counts.index, y=date_counts.values, 
                              mode='lines', name='Messages per Day'),
                    row=1, col=1
                )
            
            # 2. Media Type Distribution
            if 'media_type' in self.df.columns:
                media_counts = self.df['media_type'].value_counts()
                fig.add_trace(
                    go.Pie(labels=media_counts.index, values=media_counts.values, 
                           name="Media Types"),
                    row=1, col=2
                )
            
            # 3. File Size Distribution
            if 'file_size' in self.df.columns:
                file_sizes = self.df['file_size'].dropna()
                if not file_sizes.empty:
                    fig.add_trace(
                        go.Histogram(x=file_sizes, nbinsx=50, name="File Sizes"),
                        row=2, col=1
                    )
            
            # 4. Hourly Activity Pattern
            if 'date' in self.df.columns:
                self.df['hour'] = pd.to_datetime(self.df['date']).dt.hour
                hour_counts = self.df['hour'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=hour_counts.index, y=hour_counts.values, 
                           name="Messages by Hour"),
                    row=2, col=2
                )
            
            # 5. Daily Activity Pattern
            if 'date' in self.df.columns:
                self.df['day_of_week'] = pd.to_datetime(self.df['date']).dt.day_name()
                day_counts = self.df['day_of_week'].value_counts()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = day_counts.reindex(day_order, fill_value=0)
                fig.add_trace(
                    go.Bar(x=day_counts.index, y=day_counts.values, 
                           name="Messages by Day"),
                    row=3, col=1
                )
            
            # 6. Channel Overview (if multiple channels)
            if 'channel_username' in self.df.columns:
                channel_counts = self.df['channel_username'].value_counts()
                fig.add_trace(
                    go.Bar(x=channel_counts.index, y=channel_counts.values, 
                           name="Messages by Channel"),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=dashboard_title,
                height=1200,
                showlegend=False,
                template="plotly_white"
            )
            
            # Generate HTML
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{dashboard_title}</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                
                <!-- Google Analytics -->
                <script async src="https://www.googletagmanager.com/gtag/js?id={DEFAULT_GA_MEASUREMENT_ID}"></script>
                <script>
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{DEFAULT_GA_MEASUREMENT_ID}');
                </script>
                
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
                    .stat-box {{ 
                        background: #f8f9fa; 
                        padding: 20px; 
                        border-radius: 8px; 
                        text-align: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .stat-number {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                    .stat-label {{ color: #6c757d; margin-top: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{dashboard_title}</h1>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-number">{len(self.df):,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{self.df['channel_username'].nunique() if 'channel_username' in self.df.columns else 0}</div>
                        <div class="stat-label">Channels</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{self.df['media_type'].notna().sum() if 'media_type' in self.df.columns else 0:,}</div>
                        <div class="stat-label">Media Files</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{self.df['date'].nunique() if 'date' in self.df.columns else 0:,}</div>
                        <div class="stat-label">Active Days</div>
                    </div>
                </div>
                
                <div id="dashboard">
                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                
                <script>
                    // Make charts responsive
                    window.addEventListener('resize', function() {{
                        Plotly.Plots.resize('dashboard');
                    }});

                    // Track dashboard view with Google Analytics
                    if (typeof gtag !== 'undefined') {{
                        gtag('event', 'dashboard_view', {{
                            'dashboard_name': '{dashboard_title}',
                            'total_messages': {len(self.df)},
                            'total_channels': {self.df['channel_username'].nunique() if 'channel_username' in self.df.columns else 0}
                        }});
                    }}
                </script>
            </body>
            </html>
            """
            
            return dashboard_html
            
        except ImportError:
            # Fallback if plotly is not available
            return f"""
            <html>
            <head>
                <!-- Google Analytics -->
                <script async src="https://www.googletagmanager.com/gtag/js?id={DEFAULT_GA_MEASUREMENT_ID}"></script>
                <script>
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{DEFAULT_GA_MEASUREMENT_ID}');
                </script>
            </head>
            <body>
                <h1>{dashboard_title}</h1>
                <p>Plotly is not installed. Please install it with: pip install plotly</p>
                <h2>Basic Statistics</h2>
                <ul>
                    <li>Total Messages: {len(self.df):,}</li>
                    <li>Columns: {', '.join(self.df.columns)}</li>
                </ul>
                
                <script>
                    // Track dashboard view with Google Analytics
                    if (typeof gtag !== 'undefined') {{
                        gtag('event', 'dashboard_view', {{
                            'dashboard_name': '{dashboard_title}',
                            'total_messages': {len(self.df)},
                            'fallback_mode': true
                        }});
                    }}
                </script>
            </body>
            </html>
            """
        except Exception as e:
            return f"""
            <html>
            <head>
                <!-- Google Analytics -->
                <script async src="https://www.googletagmanager.com/gtag/js?id={DEFAULT_GA_MEASUREMENT_ID}"></script>
                <script>
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{DEFAULT_GA_MEASUREMENT_ID}');
                </script>
            </head>
            <body>
                <h1>Error Generating Dashboard</h1>
                <p>Failed to generate dashboard: {str(e)}</p>
                
                <script>
                    // Track error with Google Analytics
                    if (typeof gtag !== 'undefined') {{
                        gtag('event', 'dashboard_error', {{
                            'error_message': '{str(e)}',
                            'dashboard_name': 'Error Dashboard'
                        }});
                    }}
                </script>
            </body>
            </html>
            """

    def generate_content_analysis(self) -> Dict[str, Any]:
        """Generate content analysis insights."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}
        
        insights = {}
        
        # Text analysis
        if 'text' in self.df.columns:
            text_data = self.df['text'].dropna()
            if not text_data.empty:
                # Language detection (simple Persian text detection)
                persian_chars = sum(1 for text in text_data if any('\u0600' <= char <= '\u06FF' for char in str(text)))
                insights['persian_text_percentage'] = (persian_chars / len(text_data)) * 100
                
                # URL detection
                url_count = sum(1 for text in text_data if 'http' in str(text).lower())
                insights['url_count'] = url_count
                
                # Hashtag detection
                hashtag_count = sum(1 for text in text_data if '#' in str(text))
                insights['hashtag_count'] = hashtag_count
        
        # Media content analysis
        if 'media_type' in self.df.columns:
            media_data = self.df[self.df['media_type'].notna()]
            insights['media_files_count'] = len(media_data)
            insights['media_types'] = media_data['media_type'].value_counts().to_dict()
            
            # File size analysis
            if 'file_size' in self.df.columns:
                file_sizes = media_data['file_size'].dropna()
                if not file_sizes.empty:
                    insights['total_storage_gb'] = file_sizes.sum() / (1024**3)
                    insights['avg_file_size_mb'] = file_sizes.mean() / (1024**2)
        
        return insights
    
    def generate_engagement_metrics(self) -> Dict[str, Any]:
        """Generate engagement and interaction metrics."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}
        
        metrics = {}
        
        # Views analysis
        if 'views' in self.df.columns:
            views_data = self.df['views'].dropna()
            if not views_data.empty:
                metrics['total_views'] = views_data.sum()
                metrics['avg_views'] = views_data.mean()
                metrics['max_views'] = views_data.max()
                metrics['median_views'] = views_data.median()
        
        # Forwards analysis
        if 'forwards' in self.df.columns:
            forwards_data = self.df['forwards'].dropna()
            if not forwards_data.empty:
                metrics['total_forwards'] = forwards_data.sum()
                metrics['avg_forwards'] = forwards_data.mean()
                metrics['max_forwards'] = forwards_data.max()
        
        # Replies analysis
        if 'replies' in self.df.columns:
            replies_data = self.df['replies'].dropna()
            if not replies_data.empty:
                metrics['total_replies'] = replies_data.sum()
                metrics['avg_replies'] = replies_data.mean()
                metrics['max_replies'] = replies_data.max()
        
        # Engagement rate calculation
        if 'total_views' in metrics:
            total_messages = len(self.df)
            metrics['engagement_rate'] = (metrics['total_views'] / total_messages) if total_messages > 0 else 0
        
        return metrics
    
    def generate_file_type_analysis(self) -> Dict[str, Any]:
        """Generate detailed file type and MIME type analysis."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}
        
        analysis = {}
        
        # MIME type analysis
        if 'mime_type' in self.df.columns:
            mime_data = self.df['mime_type'].dropna()
            if not mime_data.empty:
                mime_counts = mime_data.value_counts()
                analysis['mime_types'] = mime_counts.to_dict()
                
                # Group by main type
                main_types = mime_data.str.split('/').str[0].value_counts()
                analysis['main_content_types'] = main_types.to_dict()
        
        # File extension analysis
        if 'file_name' in self.df.columns:
            file_names = self.df['file_name'].dropna()
            if not file_names.empty:
                extensions = file_names.str.extract(r'\.([^.]+)$')[0].str.lower()
                ext_counts = extensions.value_counts()
                analysis['file_extensions'] = ext_counts.head(10).to_dict()
        
        # File size by type
        if 'media_type' in self.df.columns and 'file_size' in self.df.columns:
            size_by_type = self.df.groupby('media_type')['file_size'].agg(['mean', 'sum', 'count'])
            analysis['size_by_type'] = size_by_type.to_dict()
        
        return analysis
