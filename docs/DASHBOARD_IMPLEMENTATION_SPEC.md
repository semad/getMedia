# Dashboard Implementation Specification

## Overview

This document provides detailed implementation specifications for the Dashboard command, based on the Dashboard General Design, HTML prototypes, and existing configuration structure. The implementation will generate a multi-page HTML dashboard website from analysis command output.

## Implementation Requirements

### Development Environment Setup

#### **Python Dependencies**

```bash
# Core dependencies
pip install jinja2>=3.1.0
pip install click>=8.0.0
pip install pathlib2>=2.3.0  # For Python < 3.4 compatibility

# Development dependencies
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
pip install black>=22.0.0
pip install flake8>=5.0.0
```

#### **Required Python Version**

- **Minimum**: Python 3.8+
- **Recommended**: Python 3.9+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11

### File System Requirements

#### **Directory Structure**

The implementation requires the following directory structure to exist or be created:

```text
getMedia/
├── modules/                    # Dashboard modules
│   └── dashboard_processor.py   # Complete dashboard processor
├── templates/                   # HTML templates
│   └── dashboard/
│       ├── base.html            # Base template
│       ├── index.html           # Index page template
│       ├── channel.html         # Channel page template
│       ├── single.html          # Single page template
│       └── components/
│           ├── header.html      # Header component
│           ├── metrics.html     # Metrics cards
│           ├── charts.html      # Chart containers
│           └── footer.html      # Footer component
├── reports/analysis/           # Input data directory
└── reports/dashboards/html/    # Output directory
    ├── index.html              # Generated main dashboard
    ├── {channel}.html          # Generated channel pages
    ├── dashboard-standalone.html # Generated single page dashboards
    ├── dashboard/              # Static files
    │   ├── css/
    │   │   └── dashboard.css   # Generated styles
    │   └── js/
    │       └── dashboard.js    # Generated JavaScript
    └── data/
        └── dashboard-data.json # Generated analysis data
```

#### **File Permissions**

- **Read Access**: Required for analysis input files
- **Write Access**: Required for dashboard output directory
- **Execute Access**: Required for creating subdirectories

### Input Data Requirements

#### **Analysis Command Output**

The dashboard requires analysis command output in the following format:

```text
reports/analysis/
├── file_messages/
│   └── channels/
│       ├── channel1/
│       │   ├── filename_analysis.json
│       │   ├── filesize_analysis.json
│       │   ├── message_analysis.json
│       │   └── analysis_summary.json
│       └── channel2/
│           └── ...
├── db_messages/
│   └── channels/
│       └── ...
└── diff_messages/
    └── channels/
        └── ...
```

#### **JSON File Format**

Each analysis file must contain:

```json
[
  {
    "report_type": "analysis_type",
    "generated_at": "2025-01-15T14:30:25Z",
    "analysis_version": "1.0",
    "data": {
      // Analysis-specific data
    }
  }
]
```

#### **Required Fields**

- `report_type`: Type of analysis (filename_analysis, filesize_analysis, message_analysis, analysis_summary)
- `generated_at`: ISO timestamp of analysis generation
- `analysis_version`: Version of analysis format

#### **Single Page Module Data Validation**

- **Data Size Limits**: Validate embedded data doesn't exceed memory limits
- **JSON Serialization**: Ensure all data can be serialized to JSON for embedding
- **Required Fields Validation**: Check for required fields before embedding
- **Data Sanitization**: Sanitize data to prevent XSS in embedded content
- **Channel Name Validation**: Validate channel names are safe for file system use
- **Metadata Validation**: Ensure metadata contains required generation information
- **Analysis Data Validation**: Validate analysis data structure before embedding

### Browser Compatibility

#### **Supported Browsers**

- **Chrome**: Version 90+
- **Firefox**: Version 88+
- **Safari**: Version 14+
- **Edge**: Version 90+

#### **Required Features**

- **CSS Grid**: For responsive layouts
- **ES6 JavaScript**: For modern JavaScript features
- **Fetch API**: For data loading

### Performance Requirements

#### **Data Size Limits**

- **Maximum Data Points**: 10,000 per chart
- **Maximum File Size**: 5MB per HTML file
- **Maximum Channels**: 100 channels per dashboard
- **Memory Usage**: < 100MB in browser

#### **Loading Performance**

- **Index Page Load**: < 1 second
- **Channel Page Load**: < 1.5 seconds
- **Chart Rendering**: < 500ms per chart

### Security Requirements

#### **Data Privacy**

- **No Personal Data**: Dashboard does not collect or store personal information
- **Anonymous Analytics**: Google Analytics tracking is anonymous
- **Local Processing**: All data processing happens locally

#### **File Security**

- **Input Validation**: Validate all JSON input files
- **Path Sanitization**: Sanitize file paths to prevent directory traversal
- **Safe Templates**: Use Jinja2 autoescape for XSS prevention

### Single Module Requirement

#### **Consolidated Dashboard Processor**

The implementation must use a single module called `dashboard_processor.py` that contains all dashboard functionality:

- **Single File Architecture**: All dashboard logic consolidated into one module
- **Complete Functionality**: Data processing, template rendering, file generation, and single page creation
- **Simplified Maintenance**: Easier to maintain and debug with all code in one place
- **Reduced Complexity**: No need to manage multiple module dependencies

### Single Page Module Requirement

#### **Standalone Dashboard Pages**

The implementation must support generating single-page dashboard modules that can be embedded or used independently:

- **Self-Contained HTML**: Single HTML file with embedded CSS, JavaScript, and data
- **No External Dependencies**: All resources (CSS, JS, data) embedded within the HTML file
- **Portable**: Can be shared, emailed, or hosted anywhere without additional files
- **Lightweight**: Optimized for minimal file size while maintaining functionality

#### **Use Cases**

- **Email Sharing**: Send dashboard snapshots via email
- **Embedding**: Include dashboard widgets in other websites
- **Offline Viewing**: View dashboards without internet connectivity
- **Quick Reports**: Generate standalone reports for specific time periods

#### **Technical Requirements**

- **Embedded Resources**: CSS and JavaScript must be inlined within `<style>` and `<script>` tags
- **Data Embedding**: JSON data must be embedded as JavaScript variables
- **Chart.js CDN**: Use CDN for Chart.js library (with fallback for offline mode)
- **File Size Limit**: Target single-page files under 2MB

## Configuration Extensions

### Add to `config.py`

**Note**: The following constants already exist in `config.py`:

- `ANALYSIS_BASE` (line 21)
- `DASHBOARDS_DIR` (line 39)
- `DEFAULT_GA_MEASUREMENT_ID` (line 81)

**Add these new constants to `config.py`:**

```python
# Dashboard Configuration
DASHBOARD_INPUT_DIR = ANALYSIS_BASE  # "reports/analysis" (already exists)
DASHBOARD_OUTPUT_DIR = DASHBOARDS_DIR  # "reports/dashboards/html" (already exists)

# File Naming
DASHBOARD_INDEX_FILENAME = "index.html"
DASHBOARD_CSS_FILENAME = "dashboard.css"
DASHBOARD_JS_FILENAME = "dashboard.js"
DASHBOARD_DATA_FILENAME = "dashboard-data.json"

# Static File Paths
DASHBOARD_CSS_PATH = "dashboard/css"
DASHBOARD_JS_PATH = "dashboard/js"

# HTML Generation
DASHBOARD_HTML_TITLE = "Telegram Channel Analysis Dashboard"
DASHBOARD_HTML_CHARSET = "UTF-8"
DASHBOARD_HTML_VIEWPORT = "width=device-width, initial-scale=1.0"

# Data Processing
DASHBOARD_DEFAULT_CHANNELS = "all"
DASHBOARD_SUPPORTED_ANALYSIS_TYPES = ["filename", "filesize", "message"]
DASHBOARD_SUPPORTED_SOURCE_TYPES = ["file_messages", "db_messages", "diff_messages"]
DASHBOARD_MAX_CHANNEL_NAME_LENGTH = 50

# Chart Configuration
DASHBOARD_CHART_WIDTH = 800
DASHBOARD_CHART_HEIGHT = 600
DASHBOARD_MAX_DATA_POINTS = 10000
DASHBOARD_CHARTJS_CDN_URL = "https://cdn.jsdelivr.net/npm/chart.js"

# Google Analytics Configuration
DASHBOARD_GA_MEASUREMENT_ID = DEFAULT_GA_MEASUREMENT_ID  # From existing config (line 81)
DASHBOARD_GA_ENABLED = True

# Single Page Module Configuration
DASHBOARD_SINGLE_PAGE_ENABLED = True
DASHBOARD_SINGLE_PAGE_FILENAME = "dashboard-standalone.html"
DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB = 2
```

## Implementation Architecture

### Core Components

1. **Dashboard Processor** (`modules/dashboard_processor.py`)
   - Complete dashboard generation logic
   - Data processing and aggregation
   - JSON file parsing and validation
   - HTML template rendering
   - File output management
   - Single page module generation
   - Chart.js integration
   - Resource embedding (CSS, JS, data)
   - Error handling for missing data

2. **Configuration Integration** (`config.py` extensions)
   - Dashboard-specific constants
   - File path configurations
   - Chart and UI settings
   - Single page module settings

## Implementation Steps

### Step 1: Single Module Implementation

#### `modules/dashboard_processor.py`

Create a single comprehensive module that handles all dashboard functionality:

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
    DASHBOARD_SINGLE_PAGE_ENABLED, DASHBOARD_SINGLE_PAGE_FILENAME,
    DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB, TEMPLATES_DIR,
    DASHBOARD_SUPPORTED_ANALYSIS_TYPES, DASHBOARD_SUPPORTED_SOURCE_TYPES,
    DASHBOARD_MAX_CHANNEL_NAME_LENGTH, DASHBOARD_CHARTJS_CDN_URL
)

class DashboardProcessor:
    """Complete dashboard processor handling all dashboard functionality."""
    
    def __init__(self, input_dir: Optional[str] = None, 
                 output_dir: Optional[str] = None,
                 channels: Optional[str] = None,
                 single_page: bool = False,
                 verbose: bool = False):
        self.input_dir = input_dir or DASHBOARD_INPUT_DIR
        self.output_dir = Path(output_dir or DASHBOARD_OUTPUT_DIR)
        self.channels = self._parse_channels(channels)
        self.single_page = single_page
        self.verbose = verbose
        
        self.logger = self._setup_logger()
        self.template_env = self._setup_templates()
        
    def _setup_logger(self):
        logger = logging.getLogger('dashboard_processor')
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        return logger
        
    def _setup_templates(self):
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
            
            # Generate single page modules if enabled
            if self.single_page and DASHBOARD_SINGLE_PAGE_ENABLED:
                self._generate_single_pages(dashboard_data)
            
            self.logger.info(f"Dashboard generated successfully in {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Critical error during dashboard processing: {e}")
            raise
    
    # All other methods (data processing, template rendering, etc.) 
    # will be implemented in this single module...
```

### Step 2: CLI Integration

#### Add to `main.py`

```python
@cli.command()
@click.option('--input-dir', '-i', 'input_dir', 
              default=None, help='Directory containing analysis results')
@click.option('--output-dir', '-o', 'output_dir', 
              default=None, help='Directory to save generated HTML files')
@click.option('--channels', '-c', 'channels', 
              default=None, help='Comma-separated list of channels to process')
@click.option('--single-page', '-s', 'single_page', 
              is_flag=True, help='Generate single-page standalone dashboards')
@click.option('--verbose', '-v', 'verbose', 
              is_flag=True, help='Enable verbose logging output')
def dashboard(input_dir, output_dir, channels, single_page, verbose):
    """Generate HTML dashboard from analysis results."""
    setup_logging(verbose)
    
    try:
        from modules.dashboard_processor import DashboardProcessor
        
        processor = DashboardProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            channels=channels,
            single_page=single_page,
            verbose=verbose
        )
        
        processor.process()
        click.echo("Dashboard generated successfully!")
        
    except ImportError as e:
        click.echo(f"Error importing dashboard modules: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except Exception as e:
        click.echo(f"Error generating dashboard: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
```

### Step 2: Complete Module Implementation

The `dashboard_processor.py` module should include all the following methods and functionality:

#### **Data Processing Methods**

The `dashboard_processor.py` module should include these data processing methods:

- `_discover_analysis_files()` - Find all analysis JSON files
- `_load_analysis_data()` - Load and validate JSON data
- `_aggregate_channel_data()` - Combine data from multiple files
- `_extract_channel_name()` - Extract channel names from file paths
- `_sanitize_channel_name()` - Clean channel names for file system use
- `_update_summary_metrics()` - Calculate summary statistics

#### **Template Rendering Methods**

The module should include these template rendering methods:

- `_render_index_page()` - Generate main dashboard page
- `_render_channel_page()` - Generate individual channel pages
- `_render_shared_css()` - Generate CSS styles
- `_render_shared_js()` - Generate JavaScript with Google Analytics
- `_render_single_page_template()` - Generate standalone HTML pages

#### **File Generation Methods**

The module should include these file generation methods:

- `_generate_shared_files()` - Create CSS, JS, and data files
- `_generate_html_pages()` - Create all HTML pages
- `_generate_single_pages()` - Create standalone dashboard files
- `_create_empty_dashboard_data()` - Handle cases with no data

### Step 3: HTML Templates
            index_file = self.output_dir / DASHBOARD_INDEX_FILENAME
            index_file.write_text(index_content, encoding='utf-8')
            self.logger.debug(f"Generated index page: {index_file}")
            
            # Generate channel pages
            for channel_name in data['channels'].keys():
                if self.channels and channel_name not in self.channels:
                    continue
                    
                try:
                    channel_content = self.template_manager.render_channel_page(channel_name, data, ga_measurement_id)
                    channel_file = self.output_dir / f"{channel_name}.html"
                    channel_file.write_text(channel_content, encoding='utf-8')
                    self.logger.debug(f"Generated channel page: {channel_file}")
                except Exception as e:
                    self.logger.error(f"Error generating page for channel {channel_name}: {e}")
                    continue
                    
            # Generate data file
            import json
            data_file = self.output_dir / 'data' / DASHBOARD_DATA_FILENAME
            data_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            self.logger.debug(f"Generated data file: {data_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating HTML pages: {e}")
            raise
```

### Step 5: HTML Templates

**Note**: The following template files need to be created in the `templates/dashboard/` directory:

- `index.html` - Main dashboard page
- `channel.html` - Individual channel page  
- `single.html` - Single page template (optional, for Jinja2-based single pages)

**Existing templates that can be adapted**:

- `templates/dashboard_index.html` - Can be adapted for `index.html`
- `templates/dashboard_channel.html` - Can be adapted for `channel.html`

#### `templates/dashboard/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="dashboard/css/dashboard.css">
    
    <!-- Google Analytics -->
    {% if ga_measurement_id %}
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={{ ga_measurement_id }}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', '{{ ga_measurement_id }}', {
            page_title: '{{ title }}',
            page_location: window.location.href,
            custom_map: {
                'custom_parameter_1': 'dashboard_type',
                'custom_parameter_2': 'index_page'
            }
        });
    </script>
    {% endif %}
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">📊 Dashboard</div>
                <nav class="nav">
                    <a href="#" class="nav-item active">Overview</a>
                    <a href="#" class="nav-item">Channels</a>
                    <a href="#" class="nav-item">Analysis</a>
                </nav>
                <div class="header-actions">
                    <button class="btn btn-secondary">Export</button>
                    <button class="btn btn-primary">Refresh</button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main">
        <div class="container">
            <!-- Page Header -->
            <div class="page-header">
                <h1 class="page-title">Telegram Channel Analysis</h1>
                <p class="page-subtitle">Comprehensive analysis of your Telegram channels • Generated on {{ data.metadata.generated_at }}</p>
            </div>

            <!-- Overview Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-icon messages">📈</div>
                        <div class="metric-label">Total Messages</div>
                    </div>
                    <div class="metric-value">{{ data.summary.total_messages }}</div>
                </div>
                <!-- Additional metric cards... -->
            </div>

            <!-- Channel Grid -->
            <div class="section">
                <h2 class="section-title">Channel Analysis</h2>
                <div class="channels-grid">
                    {% for channel_name, channel_data in data.channels.items() %}
                    <div class="channel-card">
                        <div class="channel-header">
                            <div class="channel-icon">📚</div>
                            <div class="channel-name">{{ channel_name }}</div>
                        </div>
                        <div class="channel-stats">
                            <div class="channel-stat">
                                <div class="channel-stat-value">{{ channel_data.messages }}</div>
                                <div class="channel-stat-label">Messages</div>
                            </div>
                            <!-- Additional stats... -->
                        </div>
                        <div class="channel-actions">
                            <a href="{{ channel_name }}.html" class="btn btn-primary btn-sm">View Details</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </main>

    <script src="dashboard/js/dashboard.js"></script>
</body>
</html>
```

### Step 6: Single Page Module

#### `modules/dashboard_single.py`

```python
import json
from pathlib import Path
from typing import Dict, Any, Optional
from config import (
    DASHBOARD_SINGLE_PAGE_ENABLED, DASHBOARD_SINGLE_PAGE_FILENAME,
    DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB, DASHBOARD_EMBED_CSS,
    DASHBOARD_EMBED_JS, DASHBOARD_EMBED_DATA, DASHBOARD_USE_CHARTJS_CDN,
    DASHBOARD_CHARTJS_CDN_URL, DASHBOARD_GA_MEASUREMENT_ID, DASHBOARD_GA_ENABLED
)

class SinglePageGenerator:
    def __init__(self, template_manager, verbose: bool = False):
        self.template_manager = template_manager
        self.verbose = verbose
        
    def generate_single_page(self, data: Dict[str, Any], output_dir: Path, 
                           channel_name: Optional[str] = None) -> Path:
        """Generate a standalone single-page dashboard."""
        if not DASHBOARD_SINGLE_PAGE_ENABLED:
            raise ValueError("Single page module is disabled in configuration")
            
        # Get Google Analytics configuration
        ga_measurement_id = DASHBOARD_GA_MEASUREMENT_ID if DASHBOARD_GA_ENABLED else None
        
        # Generate embedded CSS
        embedded_css = self._get_embedded_css() if DASHBOARD_EMBED_CSS else ""
        
        # Generate embedded JavaScript
        embedded_js = self._get_embedded_js(ga_measurement_id) if DASHBOARD_EMBED_JS else ""
        
        # Generate embedded data
        embedded_data = self._get_embedded_data(data, channel_name) if DASHBOARD_EMBED_DATA else ""
        
        # Generate Chart.js CDN or embedded version
        chartjs_script = self._get_chartjs_script()
        
        # Render single page template
        single_page_content = self._render_single_page_template(
            data=data,
            channel_name=channel_name,
            embedded_css=embedded_css,
            embedded_js=embedded_js,
            embedded_data=embedded_data,
            chartjs_script=chartjs_script,
            ga_measurement_id=ga_measurement_id
        )
        
        # Write single page file
        filename = self._get_single_page_filename(channel_name)
        output_file = output_dir / filename
        output_file.write_text(single_page_content, encoding='utf-8')
        
        # Check file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        if file_size_mb > DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB:
            if self.verbose:
                print(f"Warning: Single page file size ({file_size_mb:.2f}MB) exceeds limit ({DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB}MB)")
        
        return output_file
        
    def _get_embedded_css(self) -> str:
        """Get embedded CSS content."""
        return self.template_manager.render_shared_css()
        
    def _get_embedded_js(self, ga_measurement_id: str = None) -> str:
        """Get embedded JavaScript content."""
        return self.template_manager.render_shared_js(ga_measurement_id)
        
    def _get_embedded_data(self, data: Dict[str, Any], channel_name: Optional[str] = None) -> str:
        """Get embedded data as JavaScript variable."""
        if channel_name and channel_name in data['channels']:
            # Single channel data
            embedded_data = {
                'channel_name': channel_name,
                'channel_data': data['channels'][channel_name],
                'metadata': data['metadata']
            }
        else:
            # Full dashboard data
            embedded_data = data
            
        return f"window.dashboardData = {json.dumps(embedded_data, indent=2)};"
        
    def _get_chartjs_script(self) -> str:
        """Get Chart.js script tag (CDN or embedded)."""
        if DASHBOARD_USE_CHARTJS_CDN:
            return f'<script src="{DASHBOARD_CHARTJS_CDN_URL}"></script>'
        else:
            # For offline mode, would need to embed Chart.js library
            # This would require downloading and embedding the library
            return '<!-- Chart.js embedded for offline mode -->'
            
    def _render_single_page_template(self, data: Dict[str, Any], channel_name: Optional[str],
                                   embedded_css: str, embedded_js: str, embedded_data: str,
                                   chartjs_script: str, ga_measurement_id: str = None) -> str:
        """Render the single page template with embedded resources."""
        title = f"{channel_name} - Dashboard" if channel_name else "Dashboard Overview"
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- Embedded CSS -->
    <style>
{embedded_css}
    </style>
    
    <!-- Google Analytics -->
    {'<!-- Google tag (gtag.js) -->' if ga_measurement_id else ''}
    {f'<script async src="https://www.googletagmanager.com/gtag/js?id={ga_measurement_id}"></script>' if ga_measurement_id else ''}
    {f'''<script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{ga_measurement_id}', {{
            page_title: '{title}',
            page_location: window.location.href,
            custom_map: {{
                'custom_parameter_1': 'dashboard_type',
                'custom_parameter_2': 'single_page'
            }}
        }});
    </script>''' if ga_measurement_id else ''}
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">📊 Dashboard</div>
                <nav class="nav">
                    <a href="#" class="nav-item active">Overview</a>
                    <a href="#" class="nav-item">Channels</a>
                    <a href="#" class="nav-item">Analysis</a>
                </nav>
                <div class="header-actions">
                    <button class="btn btn-secondary" onclick="handleExport('data')">Export</button>
                    <button class="btn btn-primary" onclick="window.print()">Print</button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main">
        <div class="container">
            <!-- Page Header -->
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
                <p class="page-subtitle">Standalone Dashboard • Generated on {data['metadata'].get('generated_at', 'Unknown')}</p>
            </div>

            <!-- Dashboard Content -->
            <div id="dashboard-content">
                <!-- Content will be populated by JavaScript -->
            </div>
        </div>
    </main>

    <!-- Chart.js -->
    {chartjs_script}
    
    <!-- Embedded Data -->
    <script>
{embedded_data}
    </script>
    
    <!-- Embedded JavaScript -->
    <script>
{embedded_js}
    </script>
    
    <!-- Single Page Initialization -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Initialize single page dashboard
            initializeSinglePageDashboard();
        }});
        
        function initializeSinglePageDashboard() {{
            const content = document.getElementById('dashboard-content');
            if (window.dashboardData) {{
                if (window.dashboardData.channel_name) {{
                    // Single channel view
                    renderChannelView(window.dashboardData);
                }} else {{
                    // Full dashboard view
                    renderDashboardView(window.dashboardData);
                }}
            }} else {{
                content.innerHTML = '<div class="error">No dashboard data available</div>';
            }}
        }}
        
        function renderChannelView(data) {{
            // Render single channel dashboard
            const content = document.getElementById('dashboard-content');
            content.innerHTML = `
                <div class="channel-dashboard">
                    <h2>${{data.channel_name}}</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${{data.channel_data.messages || 0}}</div>
                            <div class="metric-label">Messages</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${{data.channel_data.files || 0}}</div>
                            <div class="metric-label">Files</div>
                        </div>
                    </div>
                </div>
            `;
        }}
        
        function renderDashboardView(data) {{
            // Render full dashboard view
            const content = document.getElementById('dashboard-content');
            const channels = Object.keys(data.channels || {{}});
            
            content.innerHTML = `
                <div class="dashboard-overview">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${{data.summary?.total_messages || 0}}</div>
                            <div class="metric-label">Total Messages</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${{data.summary?.total_files || 0}}</div>
                            <div class="metric-label">Total Files</div>
                        </div>
                    </div>
                    <div class="channels-section">
                        <h2>Channels (${{channels.length}})</h2>
                        <div class="channels-grid">
                            ${{channels.map(channel => `
                                <div class="channel-card">
                                    <h3>${{channel}}</h3>
                                    <p>Messages: ${{data.channels[channel].messages || 0}}</p>
                                </div>
                            `).join('')}}
                        </div>
                    </div>
                </div>
            `;
        }}
    </script>
</body>
</html>"""
        
    def _get_single_page_filename(self, channel_name: Optional[str] = None) -> str:
        """Get filename for single page dashboard."""
        if channel_name:
            sanitized_name = channel_name.replace('@', 'at_').replace(' ', '_').lower()
            return f"dashboard-{sanitized_name}-standalone.html"
        return DASHBOARD_SINGLE_PAGE_FILENAME
```

#### Update `modules/dashboard.py` to include single page generation

```python
# Add to DashboardGenerator class
from .dashboard_single import SinglePageGenerator

class DashboardGenerator:
    def __init__(self, input_dir: Optional[str] = None, 
                 output_dir: Optional[str] = None,
                 channels: Optional[str] = None,
                 verbose: bool = False):
        # ... existing initialization ...
        self.single_page_generator = SinglePageGenerator(self.template_manager, verbose)
        
    def generate(self):
        """Generate the complete dashboard."""
        # ... existing generation logic ...
        
        # Generate single page modules if enabled
        if self.single_page and DASHBOARD_SINGLE_PAGE_ENABLED:
            self._generate_single_pages(dashboard_data)
            
    def _generate_single_pages(self, data: Dict[str, Any]):
        """Generate single page dashboard modules."""
        self.logger.info("Generating single page modules...")
        
        try:
            # Generate full dashboard single page
            self.single_page_generator.generate_single_page(data, self.output_dir)
            self.logger.debug("Generated full dashboard single page")
            
            # Generate channel-specific single pages
            for channel_name in data['channels'].keys():
                if self.channels and channel_name not in self.channels:
                    continue
                try:
                    self.single_page_generator.generate_single_page(data, self.output_dir, channel_name)
                    self.logger.debug(f"Generated single page for channel: {channel_name}")
                except Exception as e:
                    self.logger.error(f"Error generating single page for channel {channel_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error generating single page modules: {e}")
            raise
```

## Google Analytics Integration

### Tracking Events

The dashboard includes comprehensive Google Analytics 4 tracking for the following events:

#### **Page Views**

- **Event**: `page_view`
- **Parameters**:
  - `page_title`: Dashboard page title
  - `page_location`: Current URL
  - `dashboard_type`: "index" or "channel"
  - `channel_name`: Channel name (for channel pages)

#### **Chart Interactions**

- **Event**: `chart_interaction`
- **Parameters**:
  - `chart_type`: Type of chart (bar, pie, line)
  - `action`: User action (hover, click, legend_toggle)
  - `channel_name`: Associated channel
  - `label`: Combined identifier

#### **Export Actions**

- **Event**: `export_action`
- **Parameters**:
  - `export_type`: Type of export (data, chart, pdf)
  - `channel_name`: Associated channel
  - `label`: Export identifier

### Privacy Considerations

- Google Analytics is only loaded when `DASHBOARD_GA_ENABLED` is True
- No personal data is tracked, only dashboard usage patterns
- Users can disable tracking by setting the configuration flag to False
- All tracking is anonymous and aggregated

## Error Handling

### Data Processing Errors

- Skip missing analysis files with warnings
- Handle corrupted JSON gracefully
- Provide fallback for missing data sections
- Log all errors with appropriate levels

### Output Generation Errors

- Create output directories if missing
- Handle file write permission errors
- Provide meaningful error messages
- Continue processing if individual files fail

## Performance Considerations

### Data Size Limits

- Limit charts to 10,000 data points maximum
- Target HTML file size under 5MB
- Process large datasets in chunks
- Use data sampling for overview charts

### Optimization Strategies

- Minify JSON data before embedding
- Use efficient chart rendering
- Implement lazy loading for large datasets
- Cache processed data to avoid reprocessing

## Testing Strategy

### Unit Tests

```python
# tests/test_dashboard.py
import pytest
from pathlib import Path
from modules.dashboard import DashboardGenerator
from modules.dashboard_data import DashboardDataProcessor

def test_dashboard_generator_initialization():
    generator = DashboardGenerator(verbose=True)
    assert generator.input_dir is not None
    assert generator.output_dir is not None

def test_data_processor_file_discovery():
    processor = DashboardDataProcessor("test_data", verbose=True)
    files = processor.discover_analysis_files()
    assert isinstance(files, dict)
    assert 'analysis_summary' in files


```

### Integration Tests

```python
def test_end_to_end_dashboard_generation():
    # Create test data
    test_input_dir = Path("test_data/analysis")
    test_output_dir = Path("test_output/dashboard")
    
    # Generate dashboard
    generator = DashboardGenerator(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        verbose=True
    )
    generator.generate()
    
    # Verify output files
    assert (test_output_dir / "index.html").exists()
    assert (test_output_dir / "dashboard" / "css" / "dashboard.css").exists()
    assert (test_output_dir / "dashboard" / "js" / "dashboard.js").exists()
    assert (test_output_dir / "data" / "dashboard-data.json").exists()

def test_single_page_generation():
    """Test single page module generation."""
    from modules.dashboard_single import SinglePageGenerator
    from modules.dashboard_templates import DashboardTemplateManager
    
    # Create test data
    test_data = {
        'metadata': {'generated_at': '2025-01-15T14:30:25Z'},
        'summary': {'total_messages': 1000, 'total_files': 50},
        'channels': {
            'test_channel': {
                'messages': 500,
                'files': 25,
                'filename_analysis': {},
                'filesize_analysis': {},
                'message_analysis': {}
            }
        }
    }
    
    # Generate single page
    template_manager = DashboardTemplateManager()
    single_page_generator = SinglePageGenerator(template_manager, verbose=True)
    output_file = single_page_generator.generate_single_page(
        test_data, Path("test_output"), "test_channel"
    )
    
    # Verify single page file
    assert output_file.exists()
    assert output_file.suffix == '.html'
    
    # Verify file size is within limits
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    assert file_size_mb < 2  # DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB
    
    # Verify embedded content
    content = output_file.read_text(encoding='utf-8')
    assert '<style>' in content  # Embedded CSS
    assert '<script>' in content  # Embedded JavaScript
    assert 'window.dashboardData' in content  # Embedded data
    assert 'test_channel' in content  # Channel data


```

## Deployment and Usage

### Installation

1. Add dashboard modules to the project
2. Update `config.py` with dashboard constants
3. Add CLI command to `main.py`
4. Install required dependencies

### Usage Examples

```bash
# Generate dashboard with default settings
python main.py dashboard

# Generate dashboard for specific channels
python main.py dashboard -c "books,@SherwinVakiliLibrary"

# Generate with custom directories
python main.py dashboard -i /path/to/analysis -o /path/to/output

# Generate with verbose logging
python main.py dashboard -v

# Generate single-page standalone dashboards
python main.py dashboard -s

# Generate single-page dashboards for specific channels
python main.py dashboard -s -c "books,@SherwinVakiliLibrary"

# Generate both multi-page and single-page dashboards
python main.py dashboard -s -v
```

## Implementation Checklist

### 📋 **Pre-Implementation Checklist**

- [ ] Add new constants to `config.py`
- [ ] Create `templates/dashboard/` directory
- [ ] Create `index.html` and `channel.html` templates
- [ ] Add `jinja2>=3.1.0` to requirements
- [ ] Implement complete data aggregation logic
- [ ] Add Chart.js implementation to JavaScript
- [ ] Test with sample analysis data
- [ ] Verify single page generation works
- [ ] Test error handling scenarios

This implementation specification provides a complete roadmap for building the dashboard system, integrating with the existing codebase, and maintaining consistency with the established patterns and configuration structure.
