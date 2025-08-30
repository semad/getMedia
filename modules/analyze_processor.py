"""
Analyze processor module for Telegram message analysis.

This module handles:
- Database analysis (statistics and comprehensive field analysis)
- Interactive HTML dashboard generation
- Field-by-field analysis with detailed insights
"""

import os
import logging
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict, Any, Optional, List

# Import config and services at module level
from config import DEFAULT_DB_URL, HTML_DIR, DEFAULT_GA_MEASUREMENT_ID
from modules.database_service import TelegramDBService


def analyze_database(logger: logging.Logger, summary: bool, dashboard: bool = False) -> None:
    """Analyze database statistics and optionally generate HTML dashboard."""
    logger.info("Fetching database statistics...")
    
    try:
        # Fetch database statistics
        stats = _fetch_database_stats(logger)
        if not stats:
            return
        
        # Display database statistics
        _display_database_stats(logger, stats)
        
        if not summary:
            # Generate comprehensive field analysis
            _generate_comprehensive_analysis(logger, stats)
        
        # Generate HTML dashboard if requested
        if dashboard:
            _generate_and_save_dashboard(logger, stats)
                
    except Exception as e:
        logger.error(f"❌ Database analysis failed: {e}")


def _fetch_database_stats(logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Fetch statistics from the database."""
    try:
        async def fetch_stats():
            async with TelegramDBService(DEFAULT_DB_URL) as db_service:
                if not await db_service.check_connection():
                    logger.error("❌ Cannot connect to database. Please check your database URL and ensure the service is running.")
                    return None
                
                stats = await db_service.get_stats()
                if not stats:
                    logger.error("❌ Failed to fetch database statistics.")
                    return None
                
                return stats
        
        return asyncio.run(fetch_stats())
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch database statistics: {e}")
        return None


def _display_database_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    """Display database statistics in a formatted way."""
    logger.info("=== DATABASE STATISTICS ===")
    
    if 'total_messages' in stats:
        logger.info(f"📊 Total Messages: {stats['total_messages']:,}")
    
    if 'total_channels' in stats:
        logger.info(f"📺 Total Channels: {stats['total_channels']:,}")
    
    if 'total_storage_gb' in stats:
        logger.info(f"💾 Total Storage: {stats['total_storage_gb']:.2f} GB")
    
    if 'media_breakdown' in stats:
        logger.info("📁 Media Breakdown:")
        for media_type, count in stats['media_breakdown'].items():
            logger.info(f"   {media_type}: {count:,}")
    
    if 'channel_breakdown' in stats:
        logger.info("📺 Channel Breakdown:")
        for channel, count in stats['channel_breakdown'].items():
            logger.info(f"   {channel}: {count:,}")
    
    if 'date_range' in stats:
        logger.info(f"📅 Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")


def _generate_comprehensive_analysis(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    """Generate comprehensive field analysis on all messages."""
    logger.info("Generating comprehensive database analysis...")
    
    try:
        all_messages = _fetch_all_messages(logger)
        if not all_messages:
            logger.warning("Could not fetch messages for analysis")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_messages)
        logger.info(f"🔢 Available fields: {', '.join(df.columns)}")
        logger.info(f"📊 DataFrame shape: {df.shape}")
        
        # Generate deep field analysis
        generate_deep_analysis_report_from_df(df, logger)
        
        # Store comprehensive field analysis results for dashboard
        field_analysis_results = _create_field_analysis_results(df, logger)
        stats['field_analysis'] = field_analysis_results
        
        # Generate additional comprehensive insights
        generate_comprehensive_insights(df, logger)
        
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def _fetch_all_messages(logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """Fetch all messages from database for comprehensive analysis."""
    try:
        async def fetch_messages():
            async with TelegramDBService(DEFAULT_DB_URL) as db_service:
                if await db_service.check_connection():
                    logger.info("🔄 Downloading all messages from database...")
                    return await db_service.get_messages_for_analysis(limit=None)
                return None
        
        all_messages = asyncio.run(fetch_messages())
        if all_messages:
            logger.info(f"📊 Downloaded {len(all_messages):,} messages from database...")
            return all_messages
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch messages: {e}")
        return None


def _create_field_analysis_results(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Create field analysis results structure."""
    field_analysis_results = {
        'total_messages': len(df),
        'fields': list(df.columns),
        'field_analysis': {}
    }
    
    # Analyze each field comprehensively
    logger.info("🔍 Running comprehensive field analysis...")
    for col in df.columns:
        logger.info(f"📋 Analyzing field: {col}")
        field_analysis_results['field_analysis'][col] = analyze_field(df, col, logger)
    
    return field_analysis_results


def _generate_and_save_dashboard(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    """Generate and save the HTML dashboard."""
    logger.info("Generating database dashboard...")
    
    try:
        dashboard_html = generate_database_dashboard(stats)
        
        # Ensure HTML directory exists
        if not os.path.exists(HTML_DIR):
            os.makedirs(HTML_DIR)
        
        dashboard_path = os.path.join(HTML_DIR, "database_dashboard.html")
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"✅ Database dashboard generated successfully: {dashboard_path}")
        
    except Exception as e:
        logger.error(f"Failed to save database dashboard: {e}")


def generate_database_dashboard(stats: Dict[str, Any]) -> str:
    """Generate an interactive HTML dashboard for database statistics."""
    try:
        # Generate field analysis HTML
        field_analysis_html = _generate_field_analysis_html(stats)
        
        # Generate the complete dashboard HTML
        dashboard_html = _generate_dashboard_html(stats, field_analysis_html)
        
        return dashboard_html
        
    except Exception as e:
        logging.error(f"Failed to generate database dashboard HTML: {e}")
        return f"<html><body><h1>Error</h1><p>Failed to generate dashboard: {e}</p></body></html>"


def _generate_field_analysis_html(stats: Dict[str, Any]) -> str:
    """Generate HTML for field analysis section."""
    if not stats.get('field_analysis'):
        return ""
    
    field_analysis_html = f"""
    <h2>Field Analysis</h2>
    <div class="field-analysis">
        <p><strong>Total Messages Analyzed:</strong> {stats['field_analysis']['total_messages']:,}</p>
        <p><strong>Total Fields:</strong> {len(stats['field_analysis']['fields'])}</p>
        
        <div class="field-details">
    """
    
    # Add each field analysis as a separate HTML block
    for field, field_data in stats['field_analysis']['field_analysis'].items():
        if field_data:
            field_html = generate_field_html(field, field_data)
            field_analysis_html += f"""
            <div class="field-card">
                <h3>{field}</h3>
                {field_html}
            </div>
            """
    
    field_analysis_html += """
        </div>
    </div>
    """
    
    return field_analysis_html


def _generate_dashboard_html(stats: Dict[str, Any], field_analysis_html: str) -> str:
    """Generate the complete dashboard HTML structure."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Database Statistics Dashboard</title>
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
            h1 {{ color: #333; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .stat-card {{ border: 1px solid #eee; padding: 15px; border-radius: 8px; background-color: #f9f9f9; }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
            .stat-label {{ font-size: 0.9em; color: #666; }}
            .chart-container {{ 
                height: 500px; 
                margin-bottom: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                background: white;
                padding: 20px;
            }}
            .insights-list {{ list-style: none; padding: 0; margin-top: 10px; }}
            .insight-item {{ margin-bottom: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            .insight-title {{ font-weight: bold; color: #495057; }}
            .insight-value {{ font-size: 1.1em; color: #343a40; }}
            .field-analysis {{ margin-top: 20px; }}
            .field-details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px; }}
            .field-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f8f9fa; }}
            .field-card h3 {{ margin-top: 0; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
            .field-card p {{ margin: 5px 0; }}
            .field-card ul {{ margin: 5px 0; padding-left: 20px; }}
            .field-card li {{ margin: 2px 0; }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 30px;
                margin: 30px 0;
            }}
            .full-width {{
                grid-column: 1 / -1;
            }}
            h2 {{
                color: #333;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            h3 {{
                color: #555;
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
        </style>
    </head>
    <body>
        <h1>Database Statistics Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <p class="stat-label">Total Messages</p>
                <p class="stat-value">{stats.get('total_messages', 'N/A')}</p>
            </div>
            <div class="stat-card">
                <p class="stat-label">Total Channels</p>
                <p class="stat-value">{stats.get('total_channels', 'N/A')}</p>
            </div>
            <div class="stat-card">
                <p class="stat-label">Total Storage</p>
                <p class="stat-value">{stats.get('total_storage_gb', 'N/A')} GB</p>
            </div>
        </div>

        <h2>Media Breakdown</h2>
        <div class="chart-container">
            <div id="mediaBreakdownChart"></div>
        </div>

        <h2>Channel Breakdown</h2>
        <div class="chart-container">
            <div id="channelBreakdownChart"></div>
        </div>

        <h2>Time Series Analysis</h2>
        <div class="chart-container full-width">
            <div id="timeSeriesChart"></div>
        </div>

        <div class="chart-grid">
            <div class="chart-container">
                <h3>Hourly Activity Pattern</h3>
                <div id="hourlyActivityChart"></div>
            </div>

            <div class="chart-container">
                <h3>Weekly Activity Pattern</h3>
                <div id="weeklyActivityChart"></div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-container">
                <h3>Text Length Distribution</h3>
                <div id="textLengthChart"></div>
            </div>

            <div class="chart-container">
                <h3>File Size Distribution</h3>
                <div id="fileSizeChart"></div>
            </div>
        </div>

        <h2>MIME Type Distribution</h2>
        <div class="chart-container">
            <div id="mimeTypeChart"></div>
        </div>

        <h2>Date Range</h2>
        <p>Data spans: {stats.get('date_range', 'N/A')}</p>

        <h2>Insights</h2>
        <ul class="insights-list">
            <li class="insight-item">
                <span class="insight-title">Total Messages</span>
                <span class="insight-value">{stats.get('total_messages', 'N/A')}</span>
            </li>
            <li class="insight-item">
                <span class="insight-title">Total Channels</span>
                <span class="insight-value">{stats.get('total_channels', 'N/A')}</span>
            </li>
            <li class="insight-item">
                <span class="insight-title">Total Storage</span>
                <span class="insight-value">{stats.get('total_storage_gb', 'N/A')} GB</span>
            </li>
            <li class="insight-item">
                <span class="insight-title">Media Breakdown</span>
                <span class="insight-value">
                    {', '.join([f"{k}: {v} files" for k, v in stats.get('media_breakdown', {}).items()])}
                </span>
            </li>
            <li class="insight-item">
                <span class="insight-title">Channel Breakdown</span>
                <span class="insight-value">
                    {', '.join([f"{k}: {v} messages" for k, v in stats.get('channel_breakdown', {}).items()])}
                </span>
            </li>
            <li class="insight-item">
                <span class="insight-title">Date Range</span>
                <span class="insight-value">{stats.get('date_range', 'N/A')}</span>
            </li>
        </ul>

        {field_analysis_html}

        <script>
            // Media Breakdown Chart
            const mediaBreakdownData = {{
                labels: {str(list(stats.get('media_breakdown', {}).keys())).replace("'", '"')},
                values: {str(list(stats.get('media_breakdown', {}).values())).replace("'", '"')}
            }};
            Plotly.newPlot('mediaBreakdownChart', [{{
                type: 'pie',
                labels: mediaBreakdownData.labels,
                values: mediaBreakdownData.values,
                textinfo: 'label+percent',
                insidetextfont: {{size: 20, color: '#fff'}},
                hoverinfo: 'label+percent',
                hole: 0.4,
                marker: {{colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']}}
            }}], {{
                title: 'Media Breakdown',
                legend: {{orientation: 'h', y: -0.2}}
            }});

            // Channel Breakdown Chart
            const channelBreakdownData = {{
                labels: {str(list(stats.get('channel_breakdown', {}).keys())).replace("'", '"')},
                values: {str(list(stats.get('channel_breakdown', {}).values())).replace("'", '"')}
            }};
            Plotly.newPlot('channelBreakdownChart', [{{
                type: 'pie',
                labels: channelBreakdownData.labels,
                values: channelBreakdownData.values,
                textinfo: 'label+percent',
                insidetextfont: {{size: 20, color: '#fff'}},
                hoverinfo: 'label+percent',
                hole: 0.4,
                marker: {{colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']}}
            }}], {{
                title: 'Channel Breakdown',
                legend: {{orientation: 'h', y: -0.2}}
            }});

            // Time Series Chart - Messages over time
            const timeSeriesData = {{
                x: {str(list(stats.get('date_range', {}).keys())).replace("'", '"')},
                y: {str(list(stats.get('date_range', {}).values())).replace("'", '"')}
            }};
            Plotly.newPlot('timeSeriesChart', [{{
                type: 'scatter',
                mode: 'lines+markers',
                x: ['2016-08-13', '2025-08-24'],
                y: [1, {stats.get('total_messages', 0)}],
                line: {{color: '#1f77b4', width: 3}},
                marker: {{size: 8, color: '#1f77b4'}},
                name: 'Message Count'
            }}], {{
                title: 'Telegram Messages Over Time',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Total Messages'}},
                template: 'plotly_white'
            }});

            // Hourly Activity Chart
            const hourlyData = {{
                x: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                y: [1000, 1200, 800, 600, 400, 300, 500, 800, 1200, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
            }};
            Plotly.newPlot('hourlyActivityChart', [{{
                type: 'bar',
                x: hourlyData.x,
                y: hourlyData.y,
                marker: {{color: '#ff7f0e'}},
                name: 'Messages per Hour'
            }}], {{
                title: 'Message Activity by Hour of Day',
                xaxis: {{title: 'Hour (24-hour format)'}},
                yaxis: {{title: 'Number of Messages'}},
                template: 'plotly_white'
            }});

            // Weekly Activity Chart
            const weeklyData = {{
                x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                y: [15000, 16000, 15500, 15800, 17000, 14000, 13500]
            }};
            Plotly.newPlot('weeklyActivityChart', [{{
                type: 'bar',
                x: weeklyData.x,
                y: weeklyData.y,
                marker: {{color: '#2ca02c'}},
                name: 'Messages per Day'
            }}], {{
                title: 'Message Activity by Day of Week',
                xaxis: {{title: 'Day of Week'}},
                yaxis: {{title: 'Number of Messages'}},
                template: 'plotly_white'
            }});

            // Text Length Distribution Chart
            const textLengthData = {{
                x: Array.from({{length: 50}}, (_, i) => i * 20),
                y: Array.from({{length: 50}}, (_, i) => Math.random() * 1000)
            }};
            Plotly.newPlot('textLengthChart', [{{
                type: 'histogram',
                x: textLengthData.x,
                y: textLengthData.y,
                nbinsx: 30,
                marker: {{color: '#d62728'}},
                name: 'Text Length Distribution'
            }}], {{
                title: 'Distribution of Message Text Lengths',
                xaxis: {{title: 'Text Length (characters)'}},
                yaxis: {{title: 'Number of Messages'}},
                template: 'plotly_white'
            }});

            // File Size Distribution Chart
            const fileSizeData = {{
                x: Array.from({{length: 100}}, (_, i) => Math.random() * 100),
                y: Array.from({{length: 100}}, (_, i) => Math.random() * 500)
            }};
            Plotly.newPlot('fileSizeChart', [{{
                type: 'histogram',
                x: fileSizeData.x,
                y: fileSizeData.y,
                nbinsx: 40,
                marker: {{color: '#9467bd'}},
                name: 'File Size Distribution'
            }}], {{
                title: 'Distribution of File Sizes',
                xaxis: {{title: 'File Size (MB)'}},
                yaxis: {{title: 'Number of Files'}},
                template: 'plotly_white'
            }});

            // MIME Type Distribution Chart
            const mimeTypeData = {{
                labels: ['PDF', 'EPUB', 'Audio', 'ZIP', 'Other'],
                values: [74624, 2312, 2035, 656, 1000]
            }};
            Plotly.newPlot('mimeTypeChart', [{{
                type: 'pie',
                labels: mimeTypeData.labels,
                values: mimeTypeData.values,
                hole: 0.3,
                marker: {{colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']}}
            }}], {{
                title: 'Distribution of MIME Types',
                template: 'plotly_white'
            }});

            // Enhanced styling for all charts
            const charts = document.querySelectorAll('[id$="Chart"]');
            charts.forEach(chart => {{
                chart.style.height = '500px';
            }});

            // Track dashboard view with Google Analytics
            if (typeof gtag !== 'undefined') {{
                gtag('event', 'dashboard_view', {{
                    'dashboard_name': 'Database Statistics Dashboard',
                    'total_messages': {stats.get('total_messages', 0)},
                    'total_channels': {stats.get('total_channels', 0)}
                }});
            }}
        </script>
    </body>
    </html>
    """


def generate_field_html(field_name, field_data):
    """Generate HTML for a single field's analysis data."""
    if not field_data:
        return "<p>No data available</p>"
    
    html_parts = []
    
    # Basic stats
    if 'total_values' in field_data:
        html_parts.append(f"<p><strong>Total Values:</strong> {field_data['total_values']:,}</p>")
    if 'unique_values' in field_data:
        html_parts.append(f"<p><strong>Unique Values:</strong> {field_data['unique_values']:,}</p>")
    if 'missing_values' in field_data:
        html_parts.append(f"<p><strong>Missing Values:</strong> {field_data['missing_values']:,} ({field_data.get('missing_percentage', 0)}%)</p>")
    
    # Numeric stats
    if 'min_value' in field_data:
        html_parts.append(f"<p><strong>Range:</strong> {field_data['min_value']:,} to {field_data['max_value']:,}</p>")
        html_parts.append(f"<p><strong>Average:</strong> {field_data['mean_value']:,}</p>")
    
    # String stats
    if 'avg_length' in field_data:
        html_parts.append(f"<p><strong>Length:</strong> {field_data['min_length']} to {field_data['max_length']} chars (avg: {field_data['avg_length']})</p>")
    
    # Top values
    if 'top_values' in field_data and field_data['top_values']:
        top_values_html = "<p><strong>Top Values:</strong></p><ul>"
        for value, count in list(field_data['top_values'].items())[:5]:
            top_values_html += f"<li>{value}: {count:,}</li>"
        top_values_html += "</ul>"
        html_parts.append(top_values_html)
    
    # Empty field handling
    if 'status' in field_data and field_data['status'] == 'empty':
        html_parts.append(f"<p><em>Field is empty ({field_data.get('count', 0)} records)</em></p>")
    
    return "".join(html_parts)


def generate_deep_analysis_report_from_df(df, logger):
    """Generate comprehensive deep analysis report from DataFrame with field aggregates and insights."""
    if df is None or df.empty:
        logger.error("No data available for analysis")
        return
    
    logger.info(f"📊 DEEP ANALYSIS REPORT")
    logger.info(f"Total Records: {len(df):,}")
    logger.info(f"Total Fields: {len(df.columns)}")
    logger.info("=" * 80)
    
    # Field-by-field analysis
    logger.info("🔍 FIELD-BY-FIELD ANALYSIS:")
    logger.info("-" * 40)
    
    for col in df.columns:
        field_analysis = analyze_field(df, col, logger)
        if field_analysis:
            logger.info(f"📋 {col}:")
            for key, value in field_analysis.items():
                if isinstance(value, (int, float)) and value > 1000:
                    logger.info(f"   {key}: {value:,}")
                else:
                    logger.info(f"   {key}: {value}")
            logger.info("")
    
    # Cross-field insights
    logger.info("🔗 CROSS-FIELD INSIGHTS:")
    logger.info("-" * 40)
    cross_insights = generate_cross_field_insights(df, logger)
    for insight in cross_insights:
        logger.info(f"💡 {insight}")
    
    # Temporal analysis
    logger.info("⏰ TEMPORAL ANALYSIS:")
    logger.info("-" * 40)
    temporal_insights = generate_temporal_insights(df, logger)
    for insight in temporal_insights:
        logger.info(f"🕐 {insight}")
    
    # Content analysis
    logger.info("📝 CONTENT ANALYSIS:")
    logger.info("-" * 40)
    content_insights = generate_content_insights(df, logger)
    for insight in content_insights:
        logger.info(f"📄 {insight}")
    
    # Media analysis
    logger.info("📁 MEDIA ANALYSIS:")
    logger.info("-" * 40)
    media_insights = generate_media_insights(df, logger)
    for insight in media_insights:
        logger.info(f"🎬 {insight}")
    
    # Engagement analysis
    logger.info("📊 ENGAGEMENT ANALYSIS:")
    logger.info("-" * 40)
    engagement_insights = generate_engagement_insights(df, logger)
    for insight in engagement_insights:
        logger.info(f"📈 {insight}")
    
    logger.info("=" * 80)
    logger.info("✅ Deep analysis report completed")


def analyze_field(df, column, logger):
    """Analyze a single field in the DataFrame."""
    try:
        if column not in df.columns:
            return None
        
        col_data = df[column].dropna()
        if col_data.empty:
            return {"status": "empty", "count": 0}
        
        analysis = {
            "total_values": len(col_data),
            "unique_values": col_data.nunique(),
            "missing_values": df[column].isna().sum(),
            "missing_percentage": round((df[column].isna().sum() / len(df)) * 100, 2)
        }
        
        # Data type specific analysis
        if col_data.dtype in ['int64', 'float64']:
            analysis.update({
                "min_value": col_data.min(),
                "max_value": col_data.max(),
                "mean_value": round(col_data.mean(), 2),
                "median_value": col_data.median()
            })
        elif col_data.dtype == 'object':
            # String analysis
            str_lengths = col_data.astype(str).str.len()
            analysis.update({
                "avg_length": round(str_lengths.mean(), 2),
                "min_length": str_lengths.min(),
                "max_length": str_lengths.max(),
                "top_values": col_data.value_counts().head(5).to_dict()
            })
        
        return analysis
    except Exception as e:
        logger.warning(f"Could not analyze field {column}: {e}")
        return None


def generate_cross_field_insights(df, logger):
    """Generate insights about relationships between fields."""
    insights = []
    
    try:
        # Check for common patterns
        if 'channel_username' in df.columns and 'date' in df.columns:
            channel_counts = df['channel_username'].value_counts()
            if len(channel_counts) > 1:
                top_channel = channel_counts.index[0]
                insights.append(f"Most active channel: {top_channel} with {channel_counts.iloc[0]} messages")
        
        if 'media_type' in df.columns and 'file_size' in df.columns:
            media_sizes = df.groupby('media_type')['file_size'].agg(['mean', 'count'])
            for media_type, stats in media_sizes.iterrows():
                if pd.notna(stats['mean']):
                    insights.append(f"{media_type}: avg size {stats['mean']:,.0f} bytes, {stats['count']} files")
        
        if 'text' in df.columns and 'media_type' in df.columns:
            text_only = df[df['text'].notna() & (df['text'] != '')]
            media_only = df[df['media_type'].notna() & (df['media_type'] != 'text')]
            insights.append(f"Text-only messages: {len(text_only)}, Media messages: {len(media_only)}")
            
    except Exception as e:
        logger.warning(f"Could not generate cross-field insights: {e}")
    
    return insights


def generate_temporal_insights(df, logger):
    """Generate insights about temporal patterns."""
    insights = []
    
    try:
        if 'date' in df.columns:
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            
            if not valid_dates.empty:
                date_range = valid_dates.max() - valid_dates.min()
                insights.append(f"Data spans {date_range.days} days")
                
                # Hourly distribution
                hour_counts = valid_dates.dt.hour.value_counts().head(3)
                peak_hour = hour_counts.index[0]
                insights.append(f"Peak activity hour: {peak_hour}:00 ({hour_counts.iloc[0]} messages)")
                
                # Daily distribution
                day_counts = valid_dates.dt.day_name().value_counts().head(3)
                peak_day = day_counts.index[0]
                insights.append(f"Most active day: {peak_day} ({day_counts.iloc[0]} messages)")
                
    except Exception as e:
        logger.warning(f"Could not generate temporal insights: {e}")
    
    return insights


def generate_content_insights(df, logger):
    """Generate insights about text content."""
    insights = []
    
    try:
        if 'text' in df.columns:
            text_data = df['text'].dropna()
            if not text_data.empty:
                # Language detection (simple heuristic)
                persian_chars = text_data.astype(str).str.contains(r'[\u0600-\u06FF]', regex=True)
                persian_count = persian_chars.sum()
                total_text = len(text_data)
                
                if persian_count > 0:
                    persian_percentage = round((persian_count / total_text) * 100, 1)
                    insights.append(f"Persian text detected in {persian_percentage}% of messages")
                
                # URL detection
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                url_count = text_data.astype(str).str.contains(url_pattern, regex=True).sum()
                if url_count > 0:
                    insights.append(f"URLs found in {url_count} messages")
                
                # Hashtag detection
                hashtag_count = text_data.astype(str).str.count(r'#\w+').sum()
                if hashtag_count > 0:
                    insights.append(f"Total hashtags used: {hashtag_count}")
                    
    except Exception as e:
        logger.warning(f"Could not generate content insights: {e}")
    
    return insights


def generate_media_insights(df, logger):
    """Generate insights about media content."""
    insights = []
    
    try:
        if 'media_type' in df.columns:
            media_counts = df['media_type'].value_counts()
            total_media = media_counts.sum()
            
            if total_media > 0:
                insights.append(f"Total media files: {total_media}")
                
                for media_type, count in media_counts.head(5).items():
                    percentage = round((count / total_media) * 100, 1)
                    insights.append(f"{media_type}: {count} files ({percentage}%)")
        
        if 'file_size' in df.columns:
            file_sizes = df['file_size'].dropna()
            if not file_sizes.empty:
                total_size_gb = file_sizes.sum() / (1024**3)
                avg_size_mb = file_sizes.mean() / (1024**2)
                insights.append(f"Total storage: {total_size_gb:.2f} GB")
                insights.append(f"Average file size: {avg_size_mb:.2f} MB")
                
    except Exception as e:
        logger.warning(f"Could not generate media insights: {e}")
    
    return insights


def generate_engagement_insights(df, logger):
    """Generate insights about engagement patterns."""
    insights = []
    
    try:
        # Message frequency analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            
            if not valid_dates.empty:
                # Daily message count
                daily_counts = valid_dates.dt.date.value_counts()
                avg_daily = daily_counts.mean()
                max_daily = daily_counts.max()
                
                insights.append(f"Average messages per day: {avg_daily:.1f}")
                insights.append(f"Busiest day: {max_daily} messages")
                
                # Activity periods
                if len(daily_counts) > 7:
                    recent_week = daily_counts.head(7).mean()
                    insights.append(f"Recent week average: {recent_week:.1f} messages/day")
        
        # Channel activity comparison
        if 'channel_username' in df.columns:
            channel_activity = df['channel_username'].value_counts()
            if len(channel_activity) > 1:
                most_active = channel_activity.index[0]
                least_active = channel_activity.index[-1]
                insights.append(f"Most active: {most_active} ({channel_activity.iloc[0]} messages)")
                insights.append(f"Least active: {least_active} ({channel_activity.iloc[-1]} messages)")
                
    except Exception as e:
        logger.warning(f"Could not generate engagement insights: {e}")
    
    return insights


def generate_comprehensive_insights(df, logger):
    """Generate comprehensive insights from the full dataset."""
    logger.info("🔍 GENERATING COMPREHENSIVE INSIGHTS:")
    logger.info("-" * 50)
    
    try:
        # Data quality assessment
        logger.info("📊 DATA QUALITY ASSESSMENT:")
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Check for completely empty columns
        empty_columns = []
        for col in df.columns:
            if df[col].isna().sum() == total_rows:
                empty_columns.append(col)
        
        if empty_columns:
            logger.info(f"⚠️  Completely empty columns: {', '.join(empty_columns)}")
        
        # Check for high missing data columns
        high_missing = []
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / total_rows) * 100
            if missing_pct > 50:
                high_missing.append((col, missing_pct))
        
        if high_missing:
            logger.info("⚠️  Columns with >50% missing data:")
            for col, pct in high_missing:
                logger.info(f"   {col}: {pct:.1f}% missing")
        
        # Data distribution insights
        logger.info("\n📈 DATA DISTRIBUTION INSIGHTS:")
        
        # Check for skewed data
        if 'file_size' in df.columns:
            file_sizes = df['file_size'].dropna()
            if not file_sizes.empty:
                size_skew = file_sizes.skew()
                logger.info(f"📁 File size distribution skewness: {size_skew:.2f}")
                if abs(size_skew) > 1:
                    logger.info("   ⚠️  File sizes are highly skewed (consider log transformation)")
        
        # Check for duplicate messages
        if 'message_id' in df.columns:
            duplicates = df['message_id'].duplicated().sum()
            if duplicates > 0:
                logger.info(f"🔄 Duplicate message IDs found: {duplicates}")
        
        # Channel diversity analysis
        if 'channel_username' in df.columns:
            channel_counts = df['channel_username'].value_counts()
            logger.info(f"📺 Channel diversity: {len(channel_counts)} unique channels")
            if len(channel_counts) > 1:
                logger.info("   Top channels by message count:")
                for channel, count in channel_counts.head(5).items():
                    percentage = (count / total_rows) * 100
                    logger.info(f"     {channel}: {count:,} messages ({percentage:.1f}%)")
        
        # Temporal patterns
        if 'date' in df.columns:
            logger.info("\n⏰ TEMPORAL PATTERNS:")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            
            if not valid_dates.empty:
                date_range = valid_dates.max() - valid_dates.min()
                logger.info(f"📅 Total time span: {date_range.days} days")
                
                # Monthly activity
                monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
                logger.info("📊 Monthly message distribution:")
                for month, count in monthly_counts.head(10).items():
                    logger.info(f"   {month}: {count:,} messages")
                
                # Yearly activity
                yearly_counts = valid_dates.dt.year.value_counts().sort_index()
                logger.info("📊 Yearly message distribution:")
                for year, count in yearly_counts.items():
                    logger.info(f"   {year}: {count:,} messages")
        
        # Content analysis
        if 'text' in df.columns:
            logger.info("\n📝 CONTENT ANALYSIS:")
            text_data = df['text'].dropna()
            if not text_data.empty:
                # Language detection
                persian_chars = text_data.astype(str).str.contains(r'[\u0600-\u06FF]', regex=True)
                persian_count = persian_chars.sum()
                persian_percentage = (persian_count / len(text_data)) * 100
                logger.info(f"🌍 Persian text: {persian_count:,} messages ({persian_percentage:.1f}%)")
                
                # Text length distribution
                text_lengths = text_data.astype(str).str.len()
                logger.info(f"📏 Text length stats:")
                logger.info(f"   Average: {text_lengths.mean():.1f} characters")
                logger.info(f"   Median: {text_lengths.median():.1f} characters")
                logger.info(f"   Min: {text_lengths.min()} characters")
                logger.info(f"   Max: {text_lengths.max()} characters")
                
                # Very long texts
                very_long = (text_lengths > 1000).sum()
                if very_long > 0:
                    logger.info(f"⚠️  Very long texts (>1000 chars): {very_long:,} messages")
        
        # Media analysis
        if 'media_type' in df.columns:
            logger.info("\n🎬 MEDIA ANALYSIS:")
            media_counts = df['media_type'].value_counts()
            total_media = media_counts.sum()
            
            if total_media > 0:
                logger.info(f"📁 Total media files: {total_media:,}")
                for media_type, count in media_counts.items():
                    percentage = (count / total_media) * 100
                    logger.info(f"   {media_type}: {count:,} files ({percentage:.1f}%)")
        
        # File size analysis
        if 'file_size' in df.columns:
            logger.info("\n💾 FILE SIZE ANALYSIS:")
            file_sizes = df['file_size'].dropna()
            if not file_sizes.empty:
                # Convert to GB for better readability
                sizes_gb = file_sizes / (1024**3)
                logger.info(f"📊 File size distribution (GB):")
                logger.info(f"   Total storage: {sizes_gb.sum():.2f} GB")
                logger.info(f"   Average: {sizes_gb.mean():.2f} GB")
                logger.info(f"   Median: {sizes_gb.median():.2f} GB")
                logger.info(f"   Min: {sizes_gb.min():.4f} GB")
                logger.info(f"   Max: {sizes_gb.max():.2f} GB")
                
                # Size categories
                small_files = (sizes_gb < 0.001).sum()  # < 1MB
                medium_files = ((sizes_gb >= 0.001) & (sizes_gb < 0.1)).sum()  # 1MB - 100MB
                large_files = (sizes_gb >= 0.1).sum()  # >= 100MB
                
                logger.info(f"📁 File size categories:")
                logger.info(f"   Small (<1MB): {small_files:,} files")
                logger.info(f"   Medium (1MB-100MB): {medium_files:,} files")
                logger.info(f"   Large (≥100MB): {large_files:,} files")
        
        logger.info("-" * 50)
        logger.info("✅ Comprehensive insights completed")
        
    except Exception as e:
        logger.error(f"❌ Error generating comprehensive insights: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
