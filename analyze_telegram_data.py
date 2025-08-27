#!/usr/bin/env python3
"""
Telegram Data Analysis Dashboard Generator

This script reads the exported JSON file and creates an interactive HTML dashboard
with various charts and visualizations for analyzing Telegram message data.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timezone
import click
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramDataAnalyzer:
    def __init__(self, json_file_path):
        """Initialize the analyzer with the JSON file path."""
        self.json_file_path = Path(json_file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the data from JSON file."""
        try:
            logger.info(f"Loading data from {self.json_file_path}")
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the structured JSON format with export_info and data
            if isinstance(data, dict) and 'export_info' in data:
                # Extract the actual data array
                if 'data' in data:
                    data_array = data['data']
                else:
                    # If no data key, try to find the messages array
                    for key, value in data.items():
                        if key != 'export_info' and isinstance(value, list):
                            data_array = value
                            break
                    else:
                        raise ValueError("Could not find data array in JSON file")
                
                logger.info(f"Found structured JSON with {len(data_array)} messages")
                self.df = pd.DataFrame(data_array)
            else:
                # Handle simple array format
                self.df = pd.DataFrame(data)
            
            # Convert date columns
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df['year'] = self.df['date'].dt.year
                self.df['month'] = self.df['date'].dt.month
                self.df['day_of_week'] = self.df['date'].dt.day_name()
                self.df['hour'] = self.df['date'].dt.hour
            
            # Convert numeric columns
            numeric_columns = ['message_id', 'views', 'forwards', 'replies']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Handle text length
            if 'text' in self.df.columns:
                self.df['text_length'] = self.df['text'].str.len()
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_time_series_chart(self):
        """Create time series chart of messages over time."""
        if 'date' not in self.df.columns:
            return None
        
        # Group by date and count messages
        daily_counts = self.df.groupby(self.df['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'message_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['message_count'],
            mode='lines+markers',
            name='Messages per Day',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Telegram Messages Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Messages',
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_media_distribution_chart(self):
        """Create pie chart of media types distribution."""
        if 'media_type' not in self.df.columns:
            return None
        
        media_counts = self.df['media_type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=media_counts.index,
            values=media_counts.values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title='Distribution of Media Types',
            template='plotly_white'
        )
        
        return fig
    
    def create_hourly_activity_chart(self):
        """Create bar chart of hourly activity."""
        if 'hour' not in self.df.columns:
            return None
        
        hourly_counts = self.df['hour'].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            marker_color='#ff7f0e'
        )])
        
        fig.update_layout(
            title='Message Activity by Hour of Day',
            xaxis_title='Hour (24-hour format)',
            yaxis_title='Number of Messages',
            template='plotly_white'
        )
        
        return fig
    
    def create_weekly_activity_chart(self):
        """Create bar chart of weekly activity."""
        if 'day_of_week' not in self.df.columns:
            return None
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = self.df['day_of_week'].value_counts().reindex(day_order)
        
        fig = go.Figure(data=[go.Bar(
            x=weekly_counts.index,
            y=weekly_counts.values,
            marker_color='#2ca02c'
        )])
        
        fig.update_layout(
            title='Message Activity by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Number of Messages',
            template='plotly_white'
        )
        
        return fig
    
    def create_text_length_distribution(self):
        """Create histogram of text length distribution."""
        if 'text_length' not in self.df.columns:
            return None
        
        fig = go.Figure(data=[go.Histogram(
            x=self.df['text_length'],
            nbinsx=50,
            marker_color='#d62728'
        )])
        
        fig.update_layout(
            title='Distribution of Message Text Lengths',
            xaxis_title='Text Length (characters)',
            yaxis_title='Number of Messages',
            template='plotly_white'
        )
        
        return fig
    
    def create_channel_comparison(self):
        """Create bar chart comparing channels."""
        if 'channel_username' not in self.df.columns:
            return None
        
        channel_counts = self.df['channel_username'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=channel_counts.index,
            y=channel_counts.values,
            marker_color='#9467bd'
        )])
        
        fig.update_layout(
            title='Messages by Channel',
            xaxis_title='Channel Username',
            yaxis_title='Number of Messages',
            template='plotly_white'
        )
        
        return fig
    
    def create_forwarding_analysis(self):
        """Create analysis of forwarded vs original messages."""
        if 'is_forwarded' not in self.df.columns:
            return None
        
        forward_counts = self.df['is_forwarded'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Original Messages', 'Forwarded Messages'],
            values=forward_counts.values,
            hole=0.3,
            marker_colors=['#ff7f0e', '#1f77b4']
        )])
        
        fig.update_layout(
            title='Forwarded vs Original Messages',
            template='plotly_white'
        )
        
        return fig
    
    def create_engagement_metrics(self):
        """Create subplot of engagement metrics."""
        if not all(col in self.df.columns for col in ['views', 'forwards', 'replies']):
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Message Views', 'Message Forwards', 'Message Replies', 'Engagement Summary'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Views histogram
        fig.add_trace(
            go.Histogram(x=self.df['views'], name='Views', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # Forwards histogram
        fig.add_trace(
            go.Histogram(x=self.df['forwards'], name='Forwards', marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Replies histogram
        fig.add_trace(
            go.Histogram(x=self.df['replies'], name='Replies', marker_color='#2ca02c'),
            row=2, col=1
        )
        
        # Engagement scatter plot
        fig.add_trace(
            go.Scatter(
                x=self.df['views'],
                y=self.df['forwards'],
                mode='markers',
                name='Views vs Forwards',
                marker=dict(size=3, color='#d62728', opacity=0.6)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Engagement Metrics Analysis',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_file_size_analysis(self):
        """Create comprehensive file size analysis charts."""
        if 'file_size' not in self.df.columns:
            return None
        
        # Filter out null file sizes
        file_data = self.df[self.df['file_size'].notna()].copy()
        if len(file_data) == 0:
            return None
        
        # Convert to MB for better readability
        file_data['file_size_mb'] = file_data['file_size'] / (1024 * 1024)
        
        # Create subplots for different file size analyses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('File Size Distribution', 'File Size by Media Type', 'File Size Over Time', 'File Size vs Engagement'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. File Size Distribution (histogram)
        fig.add_trace(
            go.Histogram(
                x=file_data['file_size_mb'],
                nbinsx=50,
                name='File Size Distribution',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # 2. File Size by Media Type (box plot)
        if 'media_type' in file_data.columns:
            for media_type in file_data['media_type'].unique():
                if pd.notna(media_type):
                    media_data = file_data[file_data['media_type'] == media_type]['file_size_mb']
                    if len(media_data) > 0:
                        fig.add_trace(
                            go.Box(
                                y=media_data,
                                name=media_type,
                                boxpoints='outliers'
                            ),
                            row=1, col=2
                        )
        
        # 3. File Size Over Time (scatter)
        if 'date' in file_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=file_data['date'],
                    y=file_data['file_size_mb'],
                    mode='markers',
                    name='File Size Over Time',
                    marker=dict(size=3, color='#2ca02c', opacity=0.6)
                ),
                row=2, col=1
            )
        
        # 4. File Size vs Views (scatter)
        if 'views' in file_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=file_data['file_size_mb'],
                    y=file_data['views'],
                    mode='markers',
                    name='File Size vs Views',
                    marker=dict(size=3, color='#d62728', opacity=0.6)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='File Size Analysis',
            height=600,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="File Size (MB)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="File Size (MB)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="File Size (MB)", row=2, col=1)
        fig.update_xaxes(title_text="File Size (MB)", row=2, col=2)
        fig.update_yaxes(title_text="Views", row=2, col=2)
        
        return fig

    def create_file_size_by_media_type(self):
        """Create a chart showing file size statistics by media type."""
        if 'file_size' not in self.df.columns or 'media_type' not in self.df.columns:
            return None
        
        # Filter out null values
        file_data = self.df[self.df['file_size'].notna() & self.df['media_type'].notna()].copy()
        if len(file_data) == 0:
            return None
        
        # Convert to MB
        file_data['file_size_mb'] = file_data['file_size'] / (1024 * 1024)
        
        # Group by media type and calculate statistics
        media_stats = file_data.groupby('media_type').agg({
            'file_size_mb': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        media_stats.columns = ['count', 'mean', 'median', 'std', 'min', 'max']
        media_stats = media_stats.reset_index()
        
        # Create bar chart showing average file size by media type
        fig = go.Figure(data=[go.Bar(
            x=media_stats['media_type'],
            y=media_stats['mean'],
            text=media_stats['mean'].round(2),
            textposition='auto',
            marker_color='#9467bd'
        )])
        
        fig.update_layout(
            title='Average File Size by Media Type',
            xaxis_title='Media Type',
            yaxis_title='Average File Size (MB)',
            height=400,
            template='plotly_white'
        )
        
        return fig

    def create_user_media_analysis(self):
        """Create a chart showing users and their media contribution."""
        if 'creator_username' not in self.df.columns or 'media_type' not in self.df.columns:
            return None
        
        # Filter messages with media and valid creator usernames
        media_data = self.df[
            (self.df['media_type'].notna()) & 
            (self.df['creator_username'].notna()) & 
            (self.df['creator_username'] != '')
        ].copy()
        
        if len(media_data) == 0:
            return None
        
        # Count media contributions by user
        user_media_counts = media_data.groupby('creator_username').agg({
            'media_type': 'count',
            'file_size': ['sum', 'mean'],
            'views': 'sum',
            'forwards': 'sum',
            'replies': 'sum'
        }).round(2)
        
        # Flatten column names
        user_media_counts.columns = ['media_count', 'total_size', 'avg_size', 'total_views', 'total_forwards', 'total_replies']
        user_media_counts = user_media_counts.reset_index()
        
        # Convert file sizes to MB
        user_media_counts['total_size_mb'] = user_media_counts['total_size'] / (1024 * 1024)
        user_media_counts['avg_size_mb'] = user_media_counts['avg_size'] / (1024 * 1024)
        
        # Sort by media count (descending)
        user_media_counts = user_media_counts.sort_values('media_count', ascending=False)
        
        # Limit to top 20 users for readability
        top_users = user_media_counts.head(20)
        
        # Create bar chart showing top users by media count
        fig = go.Figure(data=[go.Bar(
            x=top_users['creator_username'],
            y=top_users['media_count'],
            text=top_users['media_count'],
            textposition='auto',
            marker_color='#17a2b8',
            hovertemplate='<b>%{x}</b><br>' +
                          'Media Count: %{y}<br>' +
                          'Total Size: %{customdata[0]:.1f} MB<br>' +
                          'Avg Size: %{customdata[1]:.2f} MB<br>' +
                          'Total Views: %{customdata[2]:,}<br>' +
                          'Total Forwards: %{customdata[3]:,}<br>' +
                          'Total Replies: %{customdata[4]:,}<extra></extra>',
            customdata=list(zip(
                top_users['total_size_mb'],
                top_users['avg_size_mb'],
                top_users['total_views'],
                top_users['total_forwards'],
                top_users['total_replies']
            ))
        )])
        
        fig.update_layout(
            title='Top Users by Media Contribution',
            xaxis_title='Username',
            yaxis_title='Number of Media Attachments',
            height=500,
            template='plotly_white',
            xaxis={'tickangle': -45}
        )
        
        return fig

    def create_user_media_table(self):
        """Create a detailed table of users and their media statistics."""
        if 'creator_username' not in self.df.columns or 'media_type' not in self.df.columns:
            return None
        
        # Filter messages with media and valid creator usernames
        media_data = self.df[
            (self.df['media_type'].notna()) & 
            (self.df['creator_username'].notna()) & 
            (self.df['creator_username'] != '')
        ].copy()
        
        if len(media_data) == 0:
            return None
        
        # Count media contributions by user
        user_media_counts = media_data.groupby('creator_username').agg({
            'media_type': 'count',
            'file_size': ['sum', 'mean'],
            'views': 'sum',
            'forwards': 'sum',
            'replies': 'sum'
        }).round(2)
        
        # Flatten column names
        user_media_counts.columns = ['Media Count', 'Total Size (bytes)', 'Avg Size (bytes)', 'Total Views', 'Total Forwards', 'Total Replies']
        user_media_counts = user_media_counts.reset_index()
        
        # Convert file sizes to MB
        user_media_counts['Total Size (MB)'] = (user_media_counts['Total Size (bytes)'] / (1024 * 1024)).round(2)
        user_media_counts['Avg Size (MB)'] = (user_media_counts['Avg Size (bytes)'] / (1024 * 1024)).round(2)
        
        # Sort by media count (descending)
        user_media_counts = user_media_counts.sort_values('Media Count', ascending=False)
        
        # Select columns for display
        display_columns = ['creator_username', 'Media Count', 'Total Size (MB)', 'Avg Size (MB)', 'Total Views', 'Total Forwards', 'Total Replies']
        display_data = user_media_counts[display_columns].copy()
        display_data.columns = ['Username', 'Media Count', 'Total Size (MB)', 'Avg Size (MB)', 'Total Views', 'Total Forwards', 'Total Replies']
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_data.columns),
                fill_color='#17a2b8',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[display_data[col] for col in display_data.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11),
                height=30
            )
        )])
        
        fig.update_layout(
            title='User Media Contribution Table',
            height=600,
            template='plotly_white'
        )
        
        return fig

    def create_summary_statistics(self):
        """Create a summary statistics table."""
        if self.df is None:
            return None
        
        # Calculate summary statistics
        stats = {
            'Total Messages': len(self.df),
            'Unique Channels': self.df['channel_username'].nunique() if 'channel_username' in self.df.columns else 'N/A',
            'Date Range': f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}" if 'date' in self.df.columns else 'N/A',
            'Messages with Media': self.df['media_type'].notna().sum() if 'media_type' in self.df.columns else 'N/A',
            'Forwarded Messages': self.df['is_forwarded'].sum() if 'is_forwarded' in self.df.columns else 'N/A',
            'Average Text Length': f"{self.df['text_length'].mean():.1f}" if 'text_length' in self.df.columns else 'N/A',
            'Total Views': f"{self.df['views'].sum():,}" if 'views' in self.df.columns else 'N/A',
            'Total Forwards': f"{self.df['forwards'].sum():,}" if 'forwards' in self.df.columns else 'N/A'
        }
        
        # Add file size statistics if available
        if 'file_size' in self.df.columns:
            file_size_stats = self.df['file_size'].dropna()
            if len(file_size_stats) > 0:
                total_size_mb = file_size_stats.sum() / (1024 * 1024)
                avg_size_mb = file_size_stats.mean() / (1024 * 1024)
                stats.update({
                    'Total File Size': f"{total_size_mb:,.1f} MB",
                    'Average File Size': f"{avg_size_mb:.2f} MB",
                    'Largest File': f"{file_size_stats.max() / (1024 * 1024):.1f} MB",
                    'Smallest File': f"{file_size_stats.min() / (1024 * 1024):.2f} MB"
                })
        
        # Add user statistics if available
        if 'creator_username' in self.df.columns:
            user_stats = self.df['creator_username'].dropna()
            if len(user_stats) > 0:
                unique_users = user_stats.nunique()
                # Count users with media
                users_with_media = self.df[
                    (self.df['creator_username'].notna()) & 
                    (self.df['media_type'].notna())
                ]['creator_username'].nunique()
                stats.update({
                    'Unique Users': unique_users,
                    'Users with Media': users_with_media
                })
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='#1f77b4',
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[[k for k in stats.keys()], [v for v in stats.values()]],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title='Summary Statistics',
            template='plotly_white'
        )
        
        return fig
    
    def generate_html_dashboard(self, output_file='telegram_analysis_dashboard.html'):
        """Generate the complete HTML dashboard."""
        logger.info("Generating HTML dashboard...")
        
        # Create all charts
        charts = {
            'Time Series': self.create_time_series_chart(),
            'Media Distribution': self.create_media_distribution_chart(),
            'Hourly Activity': self.create_hourly_activity_chart(),
            'Weekly Activity': self.create_weekly_activity_chart(),
            'Text Length Distribution': self.create_text_length_distribution(),
            'Channel Comparison': self.create_channel_comparison(),
            'Forwarding Analysis': self.create_forwarding_analysis(),
            'File Size by Media Type': self.create_file_size_by_media_type(),
            'File Size Analysis': self.create_file_size_analysis(),
            'Top Users by Media': self.create_user_media_analysis(),
            'User Media Table': self.create_user_media_table(),
            'Engagement Metrics': self.create_engagement_metrics(),
            'Summary Statistics': self.create_summary_statistics()
        }
        
        # Filter out None charts
        charts = {k: v for k, v in charts.items() if v is not None}
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Telegram Data Analysis Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .chart-container {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    box-sizing: border-box;
                }}
                .chart-title {{
                    color: #333;
                    font-size: 1.5em;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }}
                .grid-row {{
                    display: flex;
                    gap: 30px;
                    margin-bottom: 30px;
                    flex-wrap: wrap;
                }}
                .chart-container {{
                    flex: 1;
                    min-width: 500px;
                    max-width: calc(50% - 15px);
                    display: flex;
                    flex-direction: column;
                }}
                .full-width {{
                    max-width: 100%;
                    margin-bottom: 30px;
                }}
                .footer {{
                    text-align: center;
                    color: #666;
                    margin-top: 40px;
                    padding: 20px;
                    border-top: 1px solid #ddd;
                }}
                
                /* Responsive design for mobile devices */
                @media (max-width: 768px) {{
                    body {{
                        padding: 10px;
                    }}
                    .header h1 {{
                        font-size: 2em;
                    }}
                    .grid-row {{
                        flex-direction: column;
                        gap: 20px;
                    }}
                    .chart-container {{
                        min-width: 100%;
                        max-width: 100%;
                    }}
                    .full-width {{
                        margin-bottom: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📱 Telegram Data Analysis Dashboard</h1>
                <p>Comprehensive analysis of {len(self.df):,} messages</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add charts in a clean grid layout
        chart_count = 0
        current_row = []
        
        for title, chart in charts.items():
            if chart is None:
                continue
            
            # Determine if chart should be full width
            is_full_width = title in ['Time Series', 'Engagement Metrics']
            
            if is_full_width:
                # Close current row if it has charts
                if current_row:
                    html_content += '<div class="grid-row">'
                    for chart_html in current_row:
                        html_content += chart_html
                    html_content += '</div>'
                    current_row = []
                
                # Add full-width chart
                html_content += f'''
                <div class="chart-container full-width">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
            else:
                # Add chart to current row
                chart_html = f'''
                <div class="chart-container">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
                current_row.append(chart_html)
                
                # If we have 2 charts in the row, close it
                if len(current_row) == 2:
                    html_content += '<div class="grid-row">'
                    for chart_html in current_row:
                        html_content += chart_html
                    html_content += '</div>'
                    current_row = []
        
        # Close any remaining charts in the last row
        if current_row:
            html_content += '<div class="grid-row">'
            for chart_html in current_row:
                html_content += chart_html
            html_content += '</div>'
        
        # Add footer
        html_content += '''
            <div class="footer">
                <p>Dashboard generated using Python, Pandas, and Plotly</p>
                <p>Data source: Telegram Messages Export</p>
            </div>
        </body>
        </html>
        '''
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_file}")
        return output_file

@click.command()
@click.option('--input-file', '-i', default='telegram_messages_export.json', 
              help='Input JSON file path')
@click.option('--output-file', '-o', default='telegram_analysis_dashboard.html',
              help='Output HTML file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_file, output_file, verbose):
    """Generate an interactive HTML dashboard from Telegram data export."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Check if input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Create analyzer and generate dashboard
        analyzer = TelegramDataAnalyzer(input_file)
        output_path = analyzer.generate_html_dashboard(output_file)
        
        logger.info(f"✅ Dashboard generated successfully!")
        logger.info(f"📁 Output file: {output_path}")
        logger.info(f"🌐 Open the HTML file in your browser to view the dashboard")
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise

if __name__ == '__main__':
    main()
