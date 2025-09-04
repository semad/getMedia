# Dashboard Implementation Guideline

## Overview

This document provides a comprehensive step-by-step guide for implementing the Dashboard command system. It follows the Dashboard Implementation Specification and provides practical instructions, code examples, and best practices for developers.

## Prerequisites

Before starting implementation, ensure you have:

- Python 3.8+ installed
- Access to the existing `getMedia` codebase
- Understanding of the analysis command output format
- Basic knowledge of Jinja2 templating
- Familiarity with Chart.js for data visualization

## Implementation Roadmap

### Phase 1: Environment Setup

1. Add configuration constants
2. Install required dependencies
3. Create directory structure
4. Set up development environment

### Phase 2: Core Implementation

1. Implement DashboardProcessor class
2. Add CLI integration
3. Create HTML templates
4. Implement data processing methods

### Phase 3: Testing & Validation

1. Create unit tests
2. Test with sample data
3. Validate HTML output
4. Performance testing

### Phase 4: Integration & Deployment

1. Integration testing
2. Documentation updates
3. User acceptance testing
4. Production deployment

### Step 1.1: Add Configuration Constants

Add the following constants to `config.py`:

```python
# Dashboard Configuration
DASHBOARD_INPUT_DIR = os.path.join(ANALYSIS_BASE, "analysis")
DASHBOARD_OUTPUT_DIR = os.path.join(DASHBOARDS_DIR, "html")
DASHBOARD_INDEX_FILENAME = "index.html"
DASHBOARD_CSS_FILENAME = "dashboard.css"
DASHBOARD_JS_FILENAME = "dashboard.js"
DASHBOARD_DATA_FILENAME = "dashboard-data.json"

# Paths and UI
DASHBOARD_CSS_PATH = "static/css"
DASHBOARD_JS_PATH = "static/js"
DASHBOARD_HTML_TITLE = "Telegram Channel Analysis Dashboard"
DASHBOARD_HTML_CHARSET = "UTF-8"
DASHBOARD_HTML_VIEWPORT = "width=device-width, initial-scale=1.0"

# Data Processing
DASHBOARD_DEFAULT_CHANNELS = []
DASHBOARD_SUPPORTED_ANALYSIS_TYPES = [
    "filename_analysis",
    "filesize_analysis", 
    "message_analysis",
    "analysis_summary"
]
DASHBOARD_SUPPORTED_SOURCE_TYPES = [
    "file_messages",
    "db_messages",
    "diff_messages"
]
DASHBOARD_MAX_CHANNEL_NAME_LENGTH = 50

# Charts and Analytics
DASHBOARD_CHART_WIDTH = 400
DASHBOARD_CHART_HEIGHT = 300
DASHBOARD_MAX_DATA_POINTS = 10000
DASHBOARD_CHARTJS_CDN_URL = "https://cdn.jsdelivr.net/npm/chart.js"
DASHBOARD_GA_MEASUREMENT_ID = DEFAULT_GA_MEASUREMENT_ID
DASHBOARD_GA_ENABLED = True

# Template Configuration
TEMPLATES_DIR = "templates"
```

### Step 1.2: Install Dependencies

Add to `requirements.txt`:

```txt
jinja2>=3.1.0
click>=8.0.0
pathlib2>=2.3.0; python_version < "3.4"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 1.3: Create Directory Structure

Create the required directories:

```bash
mkdir -p templates/dashboard
mkdir -p reports/dashboards/html/static/css
mkdir -p reports/dashboards/html/static/js
mkdir -p reports/dashboards/html/data
```

### Step 1.4: Verify Prototypes Directory

Ensure the `./prototypes/` directory exists with HTML examples:

```bash
ls -la prototypes/
# Should show: index.html, channel-books.html, mobile-demo.html, README.md
```

### Step 1.5: Validate Configuration

Create a validation script to ensure all required constants are defined:

```python
# Create validate_config.py
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import (
        DASHBOARD_INPUT_DIR, DASHBOARD_OUTPUT_DIR,
        DASHBOARD_INDEX_FILENAME, DASHBOARD_CSS_FILENAME,
        DASHBOARD_JS_FILENAME, DASHBOARD_DATA_FILENAME,
        DASHBOARD_CSS_PATH, DASHBOARD_JS_PATH,
        DASHBOARD_GA_MEASUREMENT_ID, DASHBOARD_GA_ENABLED,
        DASHBOARD_SUPPORTED_ANALYSIS_TYPES, DASHBOARD_SUPPORTED_SOURCE_TYPES,
        DASHBOARD_MAX_CHANNEL_NAME_LENGTH, DASHBOARD_CHARTJS_CDN_URL,
        TEMPLATES_DIR
    )
    print("✅ All dashboard configuration constants are properly defined!")
except ImportError as e:
    print(f"❌ Configuration validation failed: {e}")
    print("Please ensure all dashboard constants are added to config.py")
    sys.exit(1)
```

Run the validation:

```bash
python validate_config.py
```

### Step 2.1: Create DashboardProcessor Module

Create `modules/dashboard_processor.py`:

```python
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from jinja2 import Environment, FileSystemLoader

from config import (
    DASHBOARD_INPUT_DIR, DASHBOARD_OUTPUT_DIR,
    DASHBOARD_INDEX_FILENAME, DASHBOARD_CSS_FILENAME,
    DASHBOARD_JS_FILENAME, DASHBOARD_DATA_FILENAME,
    DASHBOARD_CSS_PATH, DASHBOARD_JS_PATH,
    DASHBOARD_GA_MEASUREMENT_ID, DASHBOARD_GA_ENABLED,
    DASHBOARD_SUPPORTED_ANALYSIS_TYPES, DASHBOARD_SUPPORTED_SOURCE_TYPES,
    DASHBOARD_MAX_CHANNEL_NAME_LENGTH, DASHBOARD_CHARTJS_CDN_URL,
    TEMPLATES_DIR
)


class DashboardProcessor:
    """Complete dashboard processor handling all dashboard functionality."""

    def __init__(self, input_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 channels: Optional[str] = None,
                 verbose: bool = False):
        """Initialize the dashboard processor."""
        self.input_dir = input_dir or DASHBOARD_INPUT_DIR
        self.output_dir = Path(output_dir or DASHBOARD_OUTPUT_DIR)
        self.channels = self._parse_channels(channels)
        self.verbose = verbose

        self.logger = self._setup_logger()
        self.template_env = self._setup_templates()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the dashboard processor."""
        logger = logging.getLogger('dashboard_processor')
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _setup_templates(self) -> Environment:
        """Setup Jinja2 template environment."""
        template_dir = Path(TEMPLATES_DIR) / "dashboard"
        return Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )

    def _parse_channels(self, channels_str: Optional[str]) -> List[str]:
        """Parse comma-separated channel list."""
        if not channels_str or channels_str.lower() == 'all':
            return []
        return [c.strip() for c in channels_str.split(',')]

    def process(self):
        """Main processing method - generates complete dashboard."""
        self.logger.info("Starting dashboard processing...")

        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (self.output_dir / DASHBOARD_CSS_PATH).mkdir(parents=True, exist_ok=True)
            (self.output_dir / DASHBOARD_JS_PATH).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'data').mkdir(exist_ok=True)

            # Process data
            self.logger.info("Processing analysis data...")
            files = self._discover_analysis_files()
            dashboard_data = self._aggregate_channel_data(files)

            if not dashboard_data or not dashboard_data.get('channels'):
                self.logger.warning("No channel data found. Generating empty dashboard.")
                dashboard_data = self._create_empty_dashboard_data()

            # Generate shared files
            self._generate_shared_files()

            # Generate HTML pages
            self._generate_html_pages(dashboard_data)

            self.logger.info(f"Dashboard generated successfully in {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Critical error during dashboard processing: {e}")
            raise

    # Data Processing Methods
    def _discover_analysis_files(self) -> Dict[str, List[Path]]:
        """Find all analysis JSON files in the input directory."""
        files = {}
        input_path = Path(self.input_dir)
        
        if not input_path.exists():
            self.logger.warning(f"Input directory does not exist: {input_path}")
            return files
        
        for source_type in DASHBOARD_SUPPORTED_SOURCE_TYPES:
            source_path = input_path / source_type / "channels"
            if source_path.exists():
                files[source_type] = list(source_path.rglob("*.json"))
                self.logger.debug(f"Found {len(files[source_type])} files in {source_type}")
        
        return files

    def _load_analysis_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate JSON data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            if not isinstance(data, list) or not data:
                self.logger.warning(f"Invalid data format in {file_path}")
                return None
            
            for item in data:
                if not all(key in item for key in ['report_type', 'generated_at', 'data']):
                    self.logger.warning(f"Missing required fields in {file_path}")
                    return None
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def _extract_channel_name(self, file_path: Path) -> Optional[str]:
        """Extract channel name from file path."""
        try:
            # Path structure: .../channels/channel_name/analysis_file.json
            parts = file_path.parts
            if 'channels' in parts:
                channel_index = parts.index('channels')
                if channel_index + 1 < len(parts):
                    return parts[channel_index + 1]
        except Exception as e:
            self.logger.error(f"Error extracting channel name from {file_path}: {e}")
        return None

    def _sanitize_channel_name(self, channel_name: str) -> str:
        """Clean channel name for file system use."""
        # Remove or replace unsafe characters
        sanitized = channel_name.replace('@', 'at_').replace(' ', '_')
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
        
        # Limit length
        if len(sanitized) > DASHBOARD_MAX_CHANNEL_NAME_LENGTH:
            sanitized = sanitized[:DASHBOARD_MAX_CHANNEL_NAME_LENGTH]
        
        return sanitized

    def _aggregate_channel_data(self, files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Combine data from multiple files into dashboard structure."""
        dashboard_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_channels': 0,
                'total_files': 0
            },
            'summary': {
                'total_messages': 0,
                'total_files': 0,
                'channels_count': 0
            },
            'channels': {}
        }
        
        channel_data = {}
        
        for source_type, file_list in files.items():
            for file_path in file_list:
                channel_name = self._extract_channel_name(file_path)
                if not channel_name:
                    continue
                
                # Filter by channels if specified
                if self.channels and channel_name not in self.channels:
                    continue
                
                data = self._load_analysis_data(file_path)
                if not data:
                    continue
                
                if channel_name not in channel_data:
                    channel_data[channel_name] = {
                        'messages': 0,
                        'files': 0,
                        'filename_analysis': {},
                        'filesize_analysis': {},
                        'message_analysis': {},
                        'analysis_summary': {}
                    }
                
                # Process each report in the data
                for report in data:
                    report_type = report.get('report_type', '')
                    if report_type in DASHBOARD_SUPPORTED_ANALYSIS_TYPES:
                        channel_data[channel_name][report_type] = report.get('data', {})
                        
                        # Update summary metrics
                        if report_type == 'analysis_summary':
                            summary_data = report.get('data', {})
                            channel_data[channel_name]['messages'] = summary_data.get('total_messages', 0)
                            channel_data[channel_name]['files'] = summary_data.get('total_files', 0)
        
        dashboard_data['channels'] = channel_data
        dashboard_data['metadata']['total_channels'] = len(channel_data)
        dashboard_data['summary']['channels_count'] = len(channel_data)
        
        # Calculate total summary
        for channel_data_item in channel_data.values():
            dashboard_data['summary']['total_messages'] += channel_data_item.get('messages', 0)
            dashboard_data['summary']['total_files'] += channel_data_item.get('files', 0)
        
        return dashboard_data

    def _create_empty_dashboard_data(self) -> Dict[str, Any]:
        """Create empty dashboard data structure."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_channels': 0,
                'total_files': 0
            },
            'summary': {
                'total_messages': 0,
                'total_files': 0,
                'channels_count': 0
            },
            'channels': {}
        }

    # Template Rendering Methods
    def _render_index_page(self, data: Dict[str, Any]) -> str:
        """Generate main dashboard page."""
        try:
            template = self.template_env.get_template('index.html')
            return template.render(
                title="Telegram Channel Analysis Dashboard",
                data=data,
                ga_measurement_id=DASHBOARD_GA_MEASUREMENT_ID if DASHBOARD_GA_ENABLED else None
            )
        except Exception as e:
            self.logger.error(f"Error rendering index page: {e}")
            raise

    def _render_channel_page(self, channel_name: str, data: Dict[str, Any]) -> str:
        """Generate individual channel page."""
        try:
            template = self.template_env.get_template('channel.html')
            channel_data = data['channels'].get(channel_name, {})
            
            return template.render(
                title=f"{channel_name} - Channel Analysis",
                channel_name=channel_name,
                channel_data=channel_data,
                data=data,
                ga_measurement_id=DASHBOARD_GA_MEASUREMENT_ID if DASHBOARD_GA_ENABLED else None
            )
        except Exception as e:
            self.logger.error(f"Error rendering channel page for {channel_name}: {e}")
            raise

    def _render_shared_css(self) -> str:
        """Generate CSS styles."""
        return """
/* Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.header {
    background: #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 1rem 0;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: #007bff;
}

.nav {
    display: flex;
    gap: 2rem;
}

.nav-item {
    text-decoration: none;
    color: #666;
    font-weight: 500;
}

.nav-item.active {
    color: #007bff;
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    font-weight: 500;
}

.btn-primary {
    background: #007bff;
    color: white;
}

.btn-secondary {
    background: #6c757d;
    color: white;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #007bff;
}

.metric-label {
    color: #666;
    margin-top: 0.5rem;
}

.channels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.channel-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.channel-name {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.channel-stats {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.channel-stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #28a745;
}

.channel-stat-label {
    color: #666;
    font-size: 0.9rem;
}

.chart-section {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 2rem 0;
}

.chart-container {
    height: 400px;
    width: 100%;
}

@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        gap: 1rem;
    }
    
    .nav {
        gap: 1rem;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .channels-grid {
        grid-template-columns: 1fr;
    }
}
"""

    def _render_shared_js(self) -> str:
        """Generate JavaScript with Google Analytics."""
        return f"""
// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {{
    initializeCharts();
    initializeAnalytics();
}});

function initializeCharts() {{
    // Initialize Chart.js charts
    const chartElements = document.querySelectorAll('.chart-container');
    chartElements.forEach(element => {{
        const chartType = element.dataset.chartType;
        const chartData = JSON.parse(element.dataset.chartData || '{{}}');
        
        if (chartData && Object.keys(chartData).length > 0) {{
            new Chart(element, {{
                type: chartType,
                data: chartData,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        }}
    }});
}}

function initializeAnalytics() {{
    // Google Analytics initialization
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'dashboard_loaded', {{
            'event_category': 'dashboard',
            'event_label': 'index_page'
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
}}

function handleExport(type) {{
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'export_action', {{
            'event_category': 'dashboard',
            'event_label': type
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
    
    // Export functionality
    console.log('Export requested:', type);
}}

function handleChartInteraction(chartType, action, channelName) {{
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'chart_interaction', {{
            'event_category': 'dashboard',
            'event_label': `${{chartType}}_${{action}}_${{channelName}}`
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
}}
"""

    # File Generation Methods
    def _generate_shared_files(self):
        """Create CSS, JS, and data files."""
        try:
            # Generate CSS
            css_content = self._render_shared_css()
            css_file = self.output_dir / DASHBOARD_CSS_PATH / DASHBOARD_CSS_FILENAME
            css_file.write_text(css_content, encoding='utf-8')
            self.logger.debug(f"Generated CSS file: {css_file}")
            
            # Generate JavaScript
            js_content = self._render_shared_js()
            js_file = self.output_dir / DASHBOARD_JS_PATH / DASHBOARD_JS_FILENAME
            js_file.write_text(js_content, encoding='utf-8')
            self.logger.debug(f"Generated JavaScript file: {js_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating shared files: {e}")
            raise

    def _generate_html_pages(self, data: Dict[str, Any]):
        """Create all HTML pages."""
        try:
            # Generate index page
            index_content = self._render_index_page(data)
            index_file = self.output_dir / DASHBOARD_INDEX_FILENAME
            index_file.write_text(index_content, encoding='utf-8')
            self.logger.debug(f"Generated index page: {index_file}")
            
            # Generate channel pages
            for channel_name in data['channels'].keys():
                try:
                    channel_content = self._render_channel_page(channel_name, data)
                    sanitized_name = self._sanitize_channel_name(channel_name)
                    channel_file = self.output_dir / f"{sanitized_name}.html"
                    channel_file.write_text(channel_content, encoding='utf-8')
                    self.logger.debug(f"Generated channel page: {channel_file}")
                except Exception as e:
                    self.logger.error(f"Error generating page for channel {channel_name}: {e}")
                    continue
            
            # Generate data file
            data_file = self.output_dir / 'data' / DASHBOARD_DATA_FILENAME
            data_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            self.logger.debug(f"Generated data file: {data_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating HTML pages: {e}")
            raise
```

### Step 2.2: Add CLI Integration

Add the dashboard command to `main.py`:

```python
import click
from modules.dashboard_processor import DashboardProcessor

@cli.command()
@click.option('--input-dir', '-i', 'input_dir',
              default=None, help='Directory containing analysis results')
@click.option('--output-dir', '-o', 'output_dir',
              default=None, help='Directory to save generated HTML files')
@click.option('--channels', '-c', 'channels',
              default=None, help='Comma-separated list of channels to process')
@click.option('--verbose', '-v', 'verbose',
              is_flag=True, help='Enable verbose logging output')
def dashboard(input_dir, output_dir, channels, verbose):
    """Generate HTML dashboard from analysis results."""
    # Setup logging if setup_logging function exists, otherwise use basic logging
    try:
        from main import setup_logging
        setup_logging(verbose)
    except ImportError:
        import logging
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    
    try:
        processor = DashboardProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            channels=channels,
            verbose=verbose
        )
        
        processor.process()
        click.echo("Dashboard generated successfully!")
        
    except ImportError as e:
        click.echo(f"Error importing dashboard modules: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except FileNotFoundError as e:
        click.echo(f"Required file or directory not found: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except PermissionError as e:
        click.echo(f"Permission denied: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except Exception as e:
        click.echo(f"Error generating dashboard: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
```

### Step 2.3: Create HTML Templates

#### Create `templates/dashboard/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="static/css/dashboard.css">
    {% if ga_measurement_id %}
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={{ ga_measurement_id }}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', '{{ ga_measurement_id }}');
    </script>
    {% endif %}
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">📊 Telegram Analysis Dashboard</div>
                <nav class="nav">
                    <a href="index.html" class="nav-item active">Overview</a>
                    <a href="#" class="nav-item">Export</a>
                </nav>
            </div>
        </div>
    </header>

    <main class="container">
        <h1>Channel Analysis Overview</h1>
        
        <!-- Summary Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ data.summary.channels_count }}</div>
                <div class="metric-label">Total Channels</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(data.summary.total_messages) }}</div>
                <div class="metric-label">Total Messages</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(data.summary.total_files) }}</div>
                <div class="metric-label">Total Files</div>
            </div>
        </div>

        <!-- Channels Grid -->
        <h2>Channels</h2>
        <div class="channels-grid">
            {% for channel_name, channel_data in data.channels.items() %}
            <div class="channel-card">
                <div class="channel-name">{{ channel_name }}</div>
                <div class="channel-stats">
                    <div>
                        <div class="channel-stat-value">{{ "{:,}".format(channel_data.messages) }}</div>
                        <div class="channel-stat-label">Messages</div>
                    </div>
                    <div>
                        <div class="channel-stat-value">{{ "{:,}".format(channel_data.files) }}</div>
                        <div class="channel-stat-label">Files</div>
                    </div>
                </div>
                <a href="{{ channel_name|replace('@', 'at_')|replace(' ', '_') }}.html" 
                   class="btn btn-primary">View Details</a>
            </div>
            {% endfor %}
        </div>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="static/js/dashboard.js"></script>
</body>
</html>
```

#### Create `templates/dashboard/channel.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="static/css/dashboard.css">
    {% if ga_measurement_id %}
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={{ ga_measurement_id }}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', '{{ ga_measurement_id }}');
    </script>
    {% endif %}
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">📊 Telegram Analysis Dashboard</div>
                <nav class="nav">
                    <a href="index.html" class="nav-item">Overview</a>
                    <a href="#" class="nav-item">Export</a>
                </nav>
            </div>
        </div>
    </header>

    <main class="container">
        <h1>{{ channel_name }}</h1>
        
        <!-- Channel Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(channel_data.messages) }}</div>
                <div class="metric-label">Total Messages</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "{:,}".format(channel_data.files) }}</div>
                <div class="metric-label">Total Files</div>
            </div>
        </div>

        <!-- Charts Section -->
        {% if channel_data.filename_analysis %}
        <div class="chart-section">
            <h2>File Analysis</h2>
            <div class="chart-container" 
                 data-chart-type="pie" 
                 data-chart-data="{{ channel_data.filename_analysis|tojson }}">
            </div>
        </div>
        {% endif %}

        {% if channel_data.filesize_analysis %}
        <div class="chart-section">
            <h2>File Size Distribution</h2>
            <div class="chart-container" 
                 data-chart-type="bar" 
                 data-chart-data="{{ channel_data.filesize_analysis|tojson }}">
            </div>
        </div>
        {% endif %}

        {% if channel_data.message_analysis %}
        <div class="chart-section">
            <h2>Message Analysis</h2>
            <div class="chart-container" 
                 data-chart-type="line" 
                 data-chart-data="{{ channel_data.message_analysis|tojson }}">
            </div>
        </div>
        {% endif %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="static/js/dashboard.js"></script>
</body>
</html>
```

### Step 3.1: Create Unit Tests

Create `tests/test_dashboard_processor.py`:

```python
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.dashboard_processor import DashboardProcessor


class TestDashboardProcessor:
    """Test cases for DashboardProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test DashboardProcessor initialization."""
        processor = DashboardProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            channels="test_channel",
            verbose=True
        )
        
        assert processor.input_dir == str(self.input_dir)
        assert processor.output_dir == self.output_dir
        assert processor.channels == ["test_channel"]
        assert processor.verbose is True

    def test_parse_channels(self):
        """Test channel parsing functionality."""
        processor = DashboardProcessor()
        
        # Test single channel
        result = processor._parse_channels("test_channel")
        assert result == ["test_channel"]
        
        # Test multiple channels
        result = processor._parse_channels("channel1,channel2,channel3")
        assert result == ["channel1", "channel2", "channel3"]
        
        # Test 'all' keyword
        result = processor._parse_channels("all")
        assert result == []
        
        # Test None
        result = processor._parse_channels(None)
        assert result == []

    def test_sanitize_channel_name(self):
        """Test channel name sanitization."""
        processor = DashboardProcessor()
        
        # Test normal name
        result = processor._sanitize_channel_name("test_channel")
        assert result == "test_channel"
        
        # Test name with @ symbol
        result = processor._sanitize_channel_name("@test_channel")
        assert result == "at_test_channel"
        
        # Test name with spaces
        result = processor._sanitize_channel_name("test channel")
        assert result == "test_channel"
        
        # Test name with special characters
        result = processor._sanitize_channel_name("test@channel#123")
        assert result == "testat_channel123"

    def test_create_empty_dashboard_data(self):
        """Test empty dashboard data creation."""
        processor = DashboardProcessor()
        data = processor._create_empty_dashboard_data()
        
        assert 'metadata' in data
        assert 'summary' in data
        assert 'channels' in data
        assert data['summary']['channels_count'] == 0
        assert data['summary']['total_messages'] == 0
        assert data['summary']['total_files'] == 0

    def test_discover_analysis_files_empty(self):
        """Test file discovery with empty directory."""
        processor = DashboardProcessor(input_dir=str(self.input_dir))
        files = processor._discover_analysis_files()
        
        assert isinstance(files, dict)
        assert len(files) == 0

    def test_load_analysis_data_valid(self):
        """Test loading valid analysis data."""
        # Create test data file
        test_data = [
            {
                "report_type": "analysis_summary",
                "generated_at": "2024-01-01T00:00:00",
                "data": {"total_messages": 100, "total_files": 50}
            }
        ]
        
        test_file = self.input_dir / "test_analysis.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        processor = DashboardProcessor()
        result = processor._load_analysis_data(test_file)
        
        assert result == test_data

    def test_load_analysis_data_invalid(self):
        """Test loading invalid analysis data."""
        # Create invalid data file
        test_file = self.input_dir / "invalid.json"
        with open(test_file, 'w') as f:
            f.write("invalid json content")
        
        processor = DashboardProcessor()
        result = processor._load_analysis_data(test_file)
        
        assert result is None

    @patch('modules.dashboard_processor.DashboardProcessor._discover_analysis_files')
    @patch('modules.dashboard_processor.DashboardProcessor._generate_shared_files')
    @patch('modules.dashboard_processor.DashboardProcessor._generate_html_pages')
    def test_process_success(self, mock_html, mock_shared, mock_discover):
        """Test successful dashboard processing."""
        # Mock return values
        mock_discover.return_value = {}
        
        processor = DashboardProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
        
        # Should not raise an exception
        processor.process()
        
        # Verify methods were called
        mock_discover.assert_called_once()
        mock_shared.assert_called_once()
        mock_html.assert_called_once()

    def test_render_shared_css(self):
        """Test CSS rendering."""
        processor = DashboardProcessor()
        css = processor._render_shared_css()
        
        assert isinstance(css, str)
        assert "Dashboard Styles" in css
        assert "body" in css
        assert "container" in css

    def test_render_shared_js(self):
        """Test JavaScript rendering."""
        processor = DashboardProcessor()
        js = processor._render_shared_js()
        
        assert isinstance(js, str)
        assert "Dashboard JavaScript" in js
        assert "initializeCharts" in js
        assert "initializeAnalytics" in js


if __name__ == "__main__":
    pytest.main([__file__])
```

### Step 3.2: Test with Sample Data

Create sample analysis data for testing:

```bash
# Create test data structure
mkdir -p reports/analysis/file_messages/channels/test_channel
mkdir -p reports/analysis/db_messages/channels/test_channel2

# Create sample analysis files
cat > reports/analysis/file_messages/channels/test_channel/analysis_summary.json << 'EOF'
[
  {
    "report_type": "analysis_summary",
    "generated_at": "2024-01-01T00:00:00",
    "analysis_version": "1.0",
    "data": {
      "total_messages": 1500,
      "total_files": 75,
      "date_range": {
        "start": "2023-01-01",
        "end": "2023-12-31"
      }
    }
  }
]
EOF

cat > reports/analysis/file_messages/channels/test_channel/filename_analysis.json << 'EOF'
[
  {
    "report_type": "filename_analysis",
    "generated_at": "2024-01-01T00:00:00",
    "analysis_version": "1.0",
    "data": {
      "extensions": {
        ".pdf": 25,
        ".jpg": 30,
        ".mp4": 20
      }
    }
  }
]
EOF
```

### Step 3.3: Run Tests

```bash
# Run unit tests
python -m pytest tests/test_dashboard_processor.py -v

# Test dashboard generation
python main.py dashboard --verbose

# Verify output
ls -la reports/dashboards/html/
```

### Step 4.1: Integration Testing

Test the complete workflow:

```bash
# 1. Run analysis command first
python main.py analysis --input-dir /path/to/telegram/data

# 2. Generate dashboard
python main.py dashboard --verbose

# 3. Verify output structure
tree reports/dashboards/html/
```

### Step 4.2: Performance Testing

Test with large datasets:

```bash
# Test with large number of channels
python main.py dashboard --channels "channel1,channel2,channel3,channel4,channel5" --verbose

# Monitor memory usage
python -m memory_profiler main.py dashboard --verbose
```

### Step 4.3: Browser Testing

Test generated HTML in different browsers:

1. Open `reports/dashboards/html/index.html` in Chrome
2. Test responsive design on mobile devices
3. Verify Chart.js functionality
4. Test Google Analytics integration

### Step 4.4: Documentation Updates

Update project documentation:

1. Add dashboard command to main README
2. Update CLI help documentation
3. Add usage examples
4. Document configuration options

## Troubleshooting Guide

### Common Issues

#### Issue: Template Not Found Error

```text
jinja2.exceptions.TemplateNotFound: index.html
```

**Solution:**

- Ensure `templates/dashboard/` directory exists
- Verify template files are in the correct location
- Check `TEMPLATES_DIR` configuration

#### Issue: Import Error

```text
ImportError: No module named 'modules.dashboard_processor'
```

**Solution:**

- Ensure `modules/__init__.py` exists
- Check Python path configuration
- Verify module is in correct directory

#### Issue: Permission Denied

```text
PermissionError: [Errno 13] Permission denied: '/path/to/output'
```

**Solution:**

- Check directory permissions
- Ensure write access to output directory
- Run with appropriate user permissions

#### Issue: Empty Dashboard

```text
No channel data found. Generating empty dashboard.
```

**Solution:**

- Verify analysis command has been run
- Check input directory path
- Ensure analysis files exist and are valid JSON

#### Issue: Configuration Error

```text
NameError: name 'DASHBOARD_INPUT_DIR' is not defined
```

**Solution:**

- Ensure all dashboard constants are added to `config.py`
- Check that `TEMPLATES_DIR` is defined
- Verify imports in `dashboard_processor.py`

#### Issue: Template Rendering Error

```text
jinja2.exceptions.UndefinedError: 'data' is undefined
```

**Solution:**

- Check template syntax in HTML files
- Ensure data is passed correctly to templates
- Verify Jinja2 template variables match data structure

#### Issue: Chart.js Not Loading

```text
Chart is not defined
```

**Solution:**

- Verify Chart.js CDN URL is accessible
- Check internet connection for CDN resources
- Ensure Chart.js script is loaded before dashboard.js

### Debug Mode

Enable verbose logging for debugging:

```bash
python main.py dashboard --verbose
```

### Log Analysis

Check logs for specific issues:

```bash
# Check for errors
grep -i error logs/dashboard.log

# Check for warnings
grep -i warning logs/dashboard.log

# Check processing steps
grep -i "processing\|generated" logs/dashboard.log
```

## Best Practices

### Code Organization

1. **Single Responsibility**: Each method has a single, clear purpose
2. **Error Handling**: Comprehensive error handling with logging
3. **Configuration**: All constants in `config.py`
4. **Documentation**: Clear docstrings for all methods

### Performance Optimization

1. **Lazy Loading**: Load data only when needed
2. **Memory Management**: Process large datasets in chunks
3. **Caching**: Cache processed data when possible
4. **File I/O**: Minimize file operations

### Security Considerations

1. **Input Validation**: Validate all input data
2. **Path Sanitization**: Sanitize file paths
3. **XSS Prevention**: Use Jinja2 autoescape
4. **Data Privacy**: No personal data collection

### Testing Strategy

1. **Unit Tests**: Test individual methods
2. **Integration Tests**: Test complete workflow
3. **Performance Tests**: Test with large datasets
4. **Browser Tests**: Test in multiple browsers

## Maintenance Guidelines

### Regular Updates

1. **Dependencies**: Keep Jinja2 and Chart.js updated
2. **Templates**: Update HTML templates for new features
3. **Configuration**: Review and update constants
4. **Documentation**: Keep documentation current

### Monitoring

1. **Error Logs**: Monitor for errors and warnings
2. **Performance**: Track processing times
3. **Usage**: Monitor dashboard usage patterns
4. **Feedback**: Collect user feedback

### Backup Strategy

1. **Configuration**: Backup `config.py` changes
2. **Templates**: Version control template files
3. **Generated Output**: Regular backup of dashboard output
4. **Test Data**: Maintain test data sets

## Conclusion

This implementation guideline provides a comprehensive roadmap for implementing the dashboard system. Follow the phases sequentially, test thoroughly at each step, and refer to the troubleshooting guide for common issues.

The dashboard system will provide users with an intuitive, interactive way to visualize and analyze their Telegram channel data, making it easier to understand patterns and trends in their media collections.
