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
            height=800,
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
                    margin-bottom: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chart-title {{
                    color: #333;
                    font-size: 1.5em;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                .full-width {{
                    grid-column: 1 / -1;
                }}
                .footer {{
                    text-align: center;
                    color: #666;
                    margin-top: 40px;
                    padding: 20px;
                    border-top: 1px solid #ddd;
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
        
        # Add charts in a grid layout
        chart_count = 0
        for title, chart in charts.items():
            if chart is None:
                continue
                
            # Determine if chart should be full width
            is_full_width = title in ['Time Series', 'Engagement Metrics']
            
            if chart_count == 0 or is_full_width:
                html_content += '<div class="grid">'
            
            if is_full_width:
                html_content += f'''
                <div class="chart-container full-width">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
            else:
                html_content += f'''
                <div class="chart-container">
                    <div class="chart-title">{title}</div>
                    {chart.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
                '''
            
            chart_count += 1
            
            # Close grid if needed
            if is_full_width or chart_count % 2 == 0:
                html_content += '</div>'
                chart_count = 0
        
        # Close any remaining grid
        if chart_count > 0:
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
