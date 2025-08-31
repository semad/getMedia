"""
Dashboard Renderer for Telegram message analysis.

This service handles:
- Template loading and management
- HTML generation and rendering
- Data binding to templates
- Chart component integration
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

from jinja2 import Environment, FileSystemLoader


class DashboardRenderer:
    """Service for rendering HTML dashboards using Jinja2 templates."""
    
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self._setup_filters()
        self._setup_globals()
    
    def _setup_filters(self):
        """Setup custom Jinja2 filters."""
        self.env.filters['format_number'] = self._format_number
        self.env.filters['safe_channel_name'] = self._safe_channel_name
    
    def _setup_globals(self):
        """Setup global template variables."""
        self.env.globals['current_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _format_number(self, value: Any) -> str:
        """Format numbers with thousands separators."""
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)
    
    def _safe_channel_name(self, channel_name: str) -> str:
        """Convert channel name to safe filename."""
        return channel_name.replace('@', '').replace('/', '_').replace(' ', '_')
    
    def render_index_dashboard(self, stats: Dict[str, Any], ga_measurement_id: str) -> str:
        """Render the main index dashboard."""
        try:
            template = self.env.get_template('pages/index.html')
            
            # Add GA measurement ID to context
            context = {
                'stats': stats,
                'ga_measurement_id': ga_measurement_id
            }
            
            return template.render(**context)
            
        except Exception as e:
            self.logger.error(f"Error rendering index dashboard: {e}")
            return self._generate_fallback_html("Index Dashboard", str(e))
    
    def render_channel_dashboard(self, channel: str, stats: Dict[str, Any], ga_measurement_id: str) -> str:
        """Render a channel-specific dashboard."""
        try:
            template = self.env.get_template('pages/channel_dashboard.html')
            
            # Add GA measurement ID to context
            context = {
                'channel': channel,
                'stats': stats,
                'charts': stats.get('charts', {}),
                'ga_measurement_id': ga_measurement_id
            }
            
            return template.render(**context)
            
        except Exception as e:
            self.logger.error(f"Error rendering channel dashboard for {channel}: {e}")
            return self._generate_fallback_html(f"Channel Dashboard - {channel}", str(e))
    
    def render_database_analysis(self, stats: Dict[str, Any], ga_measurement_id: str) -> str:
        """Render the database analysis page."""
        try:
            # For now, return a simple placeholder
            # This can be expanded later with full database analysis
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Database Analysis</title>
                <meta charset="UTF-8">
                <script async src="https://www.googletagmanager.com/gtag/js?id={ga_measurement_id}"></script>
                <script>
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{ga_measurement_id}');
                </script>
            </head>
            <body>
                <h1>Database Analysis</h1>
                <p>Total Messages: {stats.get('total_messages', 'N/A')}</p>
                <p>Total Channels: {stats.get('total_channels', 'N/A')}</p>
                <a href="index.html">Back to Index</a>
            </body>
            </html>
            """
            
        except Exception as e:
            self.logger.error(f"Error rendering database analysis: {e}")
            return self._generate_fallback_html("Database Analysis", str(e))
    
    def _generate_fallback_html(self, title: str, error_message: str) -> str:
        """Generate fallback HTML when template rendering fails."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .error {{ color: red; background: #ffe6e6; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="error">
                <strong>Error:</strong> {error_message}
            </div>
            <p>Please check the logs for more details.</p>
        </body>
        </html>
        """
    
    def validate_templates(self) -> Dict[str, bool]:
        """Validate that all required templates exist and are valid."""
        required_templates = [
            'base.html',
            'pages/index.html',
            'pages/channel_dashboard.html',
            'components/charts/time_series.html',
            'components/charts/pie_chart.html',
            'components/charts/histogram.html',
            'components/charts/bar_chart.html'
        ]
        
        validation_results = {}
        
        for template_name in required_templates:
            try:
                template = self.env.get_template(template_name)
                # Only validate that template can be loaded, don't try to render without context
                validation_results[template_name] = True
            except Exception as e:
                self.logger.error(f"Template validation failed for {template_name}: {e}")
                validation_results[template_name] = False
        
        return validation_results
