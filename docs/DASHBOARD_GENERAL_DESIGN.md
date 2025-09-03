# Dashboard Command General Design

## Overview

The `dashboard` command is a data visualization tool that reads the output from the analysis command and produces a single self-contained HTML file. This HTML file contains interactive charts and visual representations of the analysis results, allowing users to view and explore their Telegram channel data analysis through a web browser without requiring a running server.

## Design Goals

1. **Self-Contained HTML Generation**: Produce a single standalone HTML file that can be opened in any web browser without server dependencies
2. **Analysis Output Integration**: Read and process JSON output files from the analysis command
3. **Basic Data Visualization**: Create simple interactive charts for essential data exploration
4. **Comprehensive Dashboard**: Generate a centralized view of all analysis results and key metrics
5. **User-Friendly Interface**: Create an intuitive web-based dashboard with tabbed navigation
6. **Lightweight Technology**: Use Chart.js for fast loading and simple implementation
7. **Portable Output**: Generate a single HTML file that can be shared and viewed offline
8. **Responsive Design**: Generate HTML that works across different devices and screen sizes

## Command Line Interface

### Basic Usage
```bash
python main.py dashboard [OPTIONS]
```

### CLI Options

#### **Core Options**
- `--input-dir, -i PATH`: Directory containing analysis results (default: from `config.py` `ANALYSIS_BASE`)
- `--output-dir, -o PATH`: Directory to save generated HTML files (default: from `config.py` `DASHBOARDS_DIR`)
- `--channels, -c TEXT`: Comma-separated list of channels to process (default: all)
- `--verbose, -v`: Enable verbose logging output
- `--help, -h`: Show help message and exit

### Usage Examples

#### **Basic Usage**
```bash
# Generate dashboard with default settings (most common use case)
python main.py dashboard

# Generate dashboard with verbose logging
python main.py dashboard -v

# Generate dashboard for specific channels
python main.py dashboard -c "books,@SherwinVakiliLibrary"

# Generate dashboard with custom directories
python main.py dashboard -i /path/to/analysis -o /path/to/dashboard

# Combined options
python main.py dashboard -c "books" -o /custom/output -v
```

#### **Integration with Existing Workflow**
```bash
# Complete workflow: collect -> analysis -> dashboard
python main.py collect -c "books" -m 100
python main.py analysis --channels "books"
python main.py dashboard  # Uses analysis results automatically
```

## Architecture & Design

### High-Level Components

```
┌─────────────────┐    ┌─────────────────────────────────────────┐    ┌─────────────────┐
│   Web Browser   │    │         Dashboard Generator             │    │   Data Sources  │
│                 │    │                                         │    │                 │
│ - HTML/CSS/JS   │◄───│ ┌─────────────┐ ┌─────────────────────┐ │◄───│ - Analysis Files│
│ - Chart.js Charts │    │ │HTML Generator│ │   Data Processor    │ │    │ - JSON Results │
│ - Interactive   │    │ │- Jinja2     │ │ - Pandas Parser     │ │    │ - Channel Data │
└─────────────────┘    │ │- Templates  │ │ - Data Aggregation  │ │    └─────────────────┘
                       │ │- Static     │ │ - Data Validation   │ │
                       │ │- Self-Cont. │ └─────────────────────┘ │
                       │ └─────────────┘                         │
                       │ ┌─────────────┐ ┌─────────────────────┐ │
                       │ │File Manager │ │   Chart.js Generator  │ │
                       │ │- Input/Output│ │ - Chart Creation    │ │
                       │ │- Templates  │ │ - HTML Export       │ │
                       │ │- Validation │ │ - Data Binding      │ │
                       │ └─────────────┘ └─────────────────────┘ │
                       └─────────────────────────────────────────┘
```

### Data Flow Architecture

1. **Data Ingestion**: Read JSON analysis results from input directory
2. **Data Processing**: Parse, filter, and aggregate analysis data using Pandas
3. **Chart Generation**: Create interactive charts using Chart.js
4. **Template Processing**: Generate HTML using Jinja2 templates with embedded Chart.js charts
5. **File Output**: Write self-contained HTML files to output directory
6. **Browser Display**: Open HTML files directly in web browser

## Design Constraints

### Technology Constraints
- **Web Technologies**: HTML5, CSS3, JavaScript (ES6+)
- **Charting Library**: Chart.js for lightweight interactive visualizations (~200KB vs Plotly's 3MB)
- **Template Engine**: Jinja2 for HTML template rendering
- **Data Processing**: Pandas for JSON data manipulation
- **Data Validation**: Pydantic for configuration and data model validation

### Dependencies
The dashboard module requires the following Python packages:

#### **Core Dependencies**
- **jinja2**: >=3.1.0 (template engine for HTML rendering)
- **pandas**: >=1.5.0 (data manipulation and analysis)
- **pydantic**: >=2.0.0 (data validation and configuration management)
- **pathlib**: Built-in (file path handling)

#### **Installation Requirements**
```bash
uv add jinja2>=3.1.0 pandas>=1.5.0 pydantic>=2.0.0
uv add --dev pytest>=7.0.0
```

**Note**: Chart.js is embedded directly in the HTML file, no Python dependency required.

## Data Integration

### Analysis Results Integration

#### **Input Data Sources**
- **Path**: `reports/analysis/` directory structure
- **Format**: JSON files from analysis command output
- **Data Types**: Filename, filesize, message analysis results
- **Channel Structure**: Individual channel directories and combined reports

#### **JSON File Structure**

Based on actual analysis output, the dashboard processes the following JSON file types:

##### **1. Analysis Summary Files (`analysis_summary.json`)**
```json
[
  {
    "report_type": "analysis_summary",
    "generated_at": "2025-09-02T20:47:10.180206",
    "analysis_version": "1.0",
    "summary": {
      "total_files_analyzed": 233,
      "total_messages_analyzed": 400,
      "total_data_size_bytes": 2991560014,
      "total_data_size_mb": 2852.97,
      "duplicate_files_found": 2,
      "duplicate_sizes_found": 2,
      "languages_detected": 3,
      "primary_language": "english",
      "overall_quality_score": 0.835,
      "analysis_completeness": 1.0
    },
    "key_metrics": {
      "filename_metrics": { /* filename analysis summary */ },
      "filesize_metrics": { /* filesize analysis summary */ },
      "message_metrics": { /* message analysis summary */ }
    },
    "recommendations": [ /* analysis recommendations */ ],
    "metadata": { /* system and config info */ }
  }
]
```

##### **2. Filename Analysis Files (`filename_analysis.json`)**
```json
[
  {
    "total_files": 233,
    "unique_filenames": 232,
    "duplicate_filenames": 2,
    "duplicate_groups": [
      {
        "filename": "@Langture.pdf",
        "count": 2,
        "file_sizes": [2777600.0, 2777600.0],
        "channels": ["@SherwinVakiliLibrary"],
        "size_variation": false
      }
    ],
    "filename_patterns": {
      "avg_length": 32,
      "min_length": 9,
      "max_length": 77,
      "files_with_special_chars": 214,
      "avg_special_chars": 14,
      "files_with_extensions": 233,
      "files_with_prefixes": 70
    },
    "quality_metrics": {
      "completeness": 1.0,
      "uniqueness": 0.9957081545,
      "length_quality": 0.8454935622,
      "cleanliness": 0.0815450644,
      "descriptiveness": 1.0,
      "overall_quality": 0.8300429185
    },
    "recommendations": [ /* filename recommendations */ ],
    "report_type": "filename_analysis",
    "generated_at": "2025-09-02T20:47:10.175413",
    "analysis_version": "1.0"
  }
]
```

##### **3. Filesize Analysis Files (`filesize_analysis.json`)**
```json
[
  {
    "total_files": 233,
    "total_size_bytes": 2991560014,
    "unique_sizes": 232,
    "duplicate_sizes": 2,
    "duplicate_size_groups": [
      {
        "size_bytes": 2777600,
        "count": 2,
        "filenames": ["@Langture.pdf", "@Langture.pdf"],
        "channels": ["@SherwinVakiliLibrary"],
        "size_mb": 2.65
      }
    ],
    "size_distribution": {
      "tiny": 0,
      "small": 22,
      "medium": 133,
      "large": 74,
      "huge": 4
    },
    "size_statistics": {
      "mean": 12839313.3648068663,
      "median": 5150092.0,
      "std": 22326941.4391787015,
      "min": 47582.0,
      "max": 159515103.0,
      "p25": 2447836.0,
      "p75": 12197288.0,
      "p90": 26106663.0000000112,
      "p95": 54367661.199999921,
      "p99": 123455695.1200001687,
      "mean_mb": 12.24,
      "median_mb": 4.91,
      "max_mb": 152.13
    },
    "potential_duplicates": [
      {
        "size_bytes": 2777600,
        "size_mb": 2.65,
        "file_count": 2,
        "files": [
          {
            "message_id": 150255,
            "filename": "@Langture.pdf",
            "channel": "@SherwinVakiliLibrary",
            "date": "2025-08-30T15:30:45.000Z",
            "mime_type": "application/pdf"
          }
        ],
        "similarity_score": 1.0
      }
    ],
    "report_type": "filesize_analysis",
    "generated_at": "2025-09-02T20:47:10.176675",
    "analysis_version": "1.0"
  }
]
```

##### **4. Message Analysis Files (`message_analysis.json`)**
```json
[
  {
    "total_messages": 400,
    "text_messages": 0,
    "media_messages": 400,
    "language_distribution": {
      "english": 311,
      "russian": 2,
      "arabic": 87
    },
    "content_patterns": {
      "hashtags": 77,
      "mentions": 104,
      "urls": 85,
      "emojis": 90,
      "questions": 27,
      "exclamations": 14,
      "avg_message_length": 168
    },
    "engagement_metrics": {
      "avg_views": 86556.4581939799,
      "median_views": 5321.0,
      "max_views": 566131.0,
      "avg_forwards": 92.6856155396,
      "median_forwards": 26.0,
      "max_forwards": 789.0,
      "avg_replies": 25.3600006104,
      "median_replies": 0.0,
      "max_replies": 388.0,
      "engagement_rate": 0.0003353505
    },
    "temporal_patterns": {
      "hourly": { "2": 2, "3": 2, "4": 5, /* hourly distribution */ },
      "daily": { "Sunday": 123, "Friday": 74, /* daily distribution */ },
      "monthly": { "1": 10, "2": 5, /* monthly distribution */ }
    },
    "quality_metrics": {
      "text_completeness": 1.0,
      "media_completeness": 0.5825,
      "engagement_completeness": 0.7475,
      "avg_content_length": 168.2875,
      "content_quality": 0.395,
      "overall_quality": 0.6845
    },
    "report_type": "message_analysis",
    "generated_at": "2025-09-02T20:47:10.178689",
    "analysis_version": "1.0"
  }
]
```

### Data Processing Pipeline

#### **Data Loading**
1. **File Discovery**: Scan input directory for analysis result files
   - Discover source types: `file_messages/`, `db_messages/`, `diff_messages/`
   - Discover channels: Individual channel directories and combined reports
   - Discover analysis types: `filename_analysis.json`, `filesize_analysis.json`, `message_analysis.json`, `analysis_summary.json`

2. **JSON Parsing**: Parse JSON files and extract analysis data
   - Use `pd.read_json()` for all JSON file operations
   - Handle array-wrapped JSON objects (all files are arrays with single objects)
   - Extract nested data structures for chart generation

3. **Data Validation**: Validate data integrity and completeness
   - Check required fields: `report_type`, `generated_at`, `analysis_version`
   - Validate numeric data types and ranges
   - Check for missing or corrupted data sections

4. **Data Aggregation**: Combine data from multiple sources and channels
   - Merge data from different source types (file, API, diff)
   - Combine channel-specific data for multi-channel views
   - Aggregate metrics across analysis types

#### **Data Processing**
1. **Channel Filtering**: Filter data by specified channels
   - Filter by channel names from CLI `--channels` option
   - Handle channel name sanitization for file paths
   - Support wildcard patterns for channel selection

2. **Analysis Type Filtering**: Filter by analysis types (filename, filesize, message)
   - Filter by analysis types from CLI `--analysis-types` option
   - Support partial analysis type selection
   - Handle missing analysis type files gracefully

3. **Date Range Filtering**: Filter data by date ranges
   - Parse date range from CLI `--date-range` option (format: YYYY-MM-DD:YYYY-MM-DD)
   - Filter temporal data in message analysis
   - Handle timezone considerations

4. **Data Transformation**: Transform data for chart visualization
   - Convert nested JSON structures to flat data for Chart.js
   - Transform temporal patterns to time-series data
   - Normalize data for consistent chart scaling
   - Prepare data for different chart types (bar, pie, line)

#### **Data Loading Implementation**

##### **Simple JSON Parsing**

The dashboard module uses direct pandas JSON parsing with basic validation:

- **File Discovery**: Scans input directory for analysis JSON files
- **JSON Loading**: Uses `pd.read_json()` for all JSON file operations
- **Data Validation**: Basic field checking and error handling
- **Data Transformation**: Simple data conversion for Chart.js visualization

##### **Data Processing Functions**

The dashboard module includes simple data processing functions that:

- **Load Analysis Files**: Read JSON files from analysis command output
- **Extract Key Metrics**: Pull summary statistics and key data points
- **Format for Charts**: Convert data to Chart.js format (labels, datasets)
- **Handle Errors**: Gracefully handle missing files and corrupted data

## Implementation Details

### Configuration Models

The dashboard module uses Pydantic V2 models for configuration management and data validation.

#### **DashboardConfig Model**

The main configuration model includes:

- **Input Configuration**: Input directory path for analysis results
- **Output Configuration**: Output directory path for generated HTML files
- **Template Configuration**: HTML template file path and customization options
- **Data Processing Configuration**: Channel filters, analysis type filters, date ranges
- **HTML Generation Configuration**: CDN usage, data inclusion, minification options
- **Logging Configuration**: Verbose mode, log levels

#### **ChartConfig Model**

Chart configuration model including:

- **Chart Types**: bar, pie, line charts
- **Dimensions**: Width, height, responsive options
- **Interactive Features**: Hover tooltips, legend
- **Color Schemes**: Chart.js default color palettes
- **Export Options**: HTML embedding only

### Configuration Factory

A factory function `create_dashboard_config()` provides easy instantiation of the DashboardConfig model with proper validation and default values.

### Configuration Constants

All constants and default values MUST be defined in `config.py` and imported where needed:

#### **File Path Constants (Use Existing)**
```python
# config.py - Use existing constants
DASHBOARD_INPUT_DIR = ANALYSIS_BASE  # "reports/analysis"
DASHBOARD_OUTPUT_DIR = DASHBOARDS_DIR  # "reports/dashboards/html"
```

#### **File Naming Constants**
```python
# config.py
DASHBOARD_FILENAME = "dashboard.html"
```

#### **Chart Configuration Constants**
```python
# config.py
DASHBOARD_CHART_WIDTH = 800
DASHBOARD_CHART_HEIGHT = 600
DASHBOARD_COLOR_PALETTE = "default"
DASHBOARD_MAX_DATA_POINTS = 10000
```

#### **HTML Generation Constants**
```python
# config.py
DASHBOARD_HTML_TITLE = "Telegram Channel Analysis Dashboard"
DASHBOARD_HTML_CHARSET = "UTF-8"
DASHBOARD_HTML_VIEWPORT = "width=device-width, initial-scale=1.0"
```

#### **Data Processing Constants**
```python
# config.py
DASHBOARD_DEFAULT_CHANNELS = "all"
DASHBOARD_SUPPORTED_ANALYSIS_TYPES = ["filename", "filesize", "message"]
DASHBOARD_SUPPORTED_SOURCE_TYPES = ["file_messages", "db_messages", "diff_messages"]
DASHBOARD_MAX_CHANNEL_NAME_LENGTH = 50
```

#### **Usage in Dashboard Module**
```python
# dashboard.py
from config import (
    DASHBOARD_INPUT_DIR,
    DASHBOARD_OUTPUT_DIR,
    DASHBOARD_CHART_WIDTH,
    DASHBOARD_CHART_HEIGHT,
    DASHBOARD_DEFAULT_CHANNELS,
    DASHBOARD_SUPPORTED_ANALYSIS_TYPES,
    DASHBOARD_SUPPORTED_SOURCE_TYPES
)

# Use constants instead of hardcoded values
input_dir = config.input_dir or DASHBOARD_INPUT_DIR
output_dir = config.output_dir or DASHBOARD_OUTPUT_DIR
channels = config.channels or DASHBOARD_DEFAULT_CHANNELS

chart_config = {
    "width": DASHBOARD_CHART_WIDTH,
    "height": DASHBOARD_CHART_HEIGHT
}
```

## File Organization Structure

### Output Directory Structure

The dashboard command generates a single comprehensive HTML file:

```
reports/dashboards/html/
└── dashboard.html                    # Complete dashboard with all data and visualizations
```

**Note**: All CSS, JavaScript, and data are embedded inline in the single HTML file for complete self-contained operation.

### File Naming Conventions

#### **HTML Files**
- **Dashboard File**: `dashboard.html` (single comprehensive file)

#### **Channel Name Sanitization (for internal data organization)**
- Replace `@` with `at_` (e.g., `@channel` → `at_channel`)
- Replace special characters with underscores
- Convert to lowercase
- Limit to 50 characters (from `config.py` `DASHBOARD_MAX_CHANNEL_NAME_LENGTH`)

### File Content Structure

#### **Dashboard File (dashboard.html)**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telegram Channel Analysis Dashboard</title>
    <style>/* Embedded CSS */</style>
</head>
<body>
    <header>
        <h1>Telegram Channel Analysis Dashboard</h1>
        <nav><!-- Channel and source navigation --></nav>
    </header>
    <main>
        <section class="overview-cards"><!-- Key metrics cards --></section>
        <section class="channel-tabs"><!-- Channel-specific data tabs --></section>
        <section class="analysis-charts"><!-- Analysis result charts --></section>
        <section class="data-summary"><!-- Summary tables --></section>
    </main>
    <footer>
        <p>Generated on: {timestamp}</p>
        <p>Data sources: {sources}</p>
    </footer>
    <script>/* Embedded charting library and custom JavaScript */</script>
</body>
</html>
```

### Data Embedding Strategy

#### **Inline Data Embedding (Default and Only Option)**
- All analysis data embedded directly in HTML files
- Chart.js JavaScript library embedded inline
- CSS styles embedded in `<style>` tags
- Custom JavaScript embedded in `<script>` tags
- **Advantage**: Completely self-contained, works offline, no external dependencies
- **Trade-off**: Larger file sizes for complete portability

### File Size Considerations

#### **Estimated File Sizes**
- **Dashboard File**: 1-3 MB (with embedded data and Chart.js)
- **Chart.js Library**: ~200KB (vs Plotly's 3MB)
- **Analysis Data**: 500KB-2MB (depending on dataset size)

#### **Optimization Strategies**
- **Data Compression**: Minify JSON data before embedding
- **Chart Optimization**: Limit data points for large datasets (from `config.py` `DASHBOARD_MAX_DATA_POINTS`)
- **Data Sampling**: Sample large datasets for overview charts
- **Library Choice**: Chart.js is 15x smaller than Plotly
- **Lazy Loading**: Load chart data only when needed
- **Caching**: Cache processed data to avoid reprocessing

### Directory Permissions

#### **Input Directory**
- **Read Access**: Required for reading analysis JSON files
- **No Write Access**: Dashboard command only reads from this directory

#### **Output Directory**
- **Write Access**: Required for creating HTML files and subdirectories
- **Read Access**: Required for overwriting existing files
- **Execute Access**: Required for creating subdirectories

### File Generation Rules

#### **Overwrite Behavior**
- **Existing Files**: Overwrite by default (generated files can be regenerated)
- **Partial Generation**: Continue if some files fail to generate

## Dashboard Interface Design

### HTML File Structure

#### **Header Section**
- **Title**: Dashboard title and analysis date range
- **Navigation**: Channel selection and data source indicators
- **Export Options**: Download data and print functionality

#### **Main Content Area**
- **Overview Cards**: Key metrics and summary statistics
- **Chart Grid**: Interactive visualizations in responsive grid layout
- **Analysis Summary**: Comprehensive analysis results display

#### **Footer Section**
- **Data Source Information**: Analysis timestamp and data sources
- **Chart.js Attribution**: Required attribution for Chart.js library

### Visualization Components

#### **Overview Metrics Cards**

The dashboard displays key metrics in card format including:

- **Total Messages**: Message count with analysis period
- **Active Channels**: Channel count and data coverage
- **Data Quality**: Quality score and completeness indicators
- **File Statistics**: File counts and size distributions

#### **Interactive Charts**

Charts are generated using Chart.js with:

- **Chart Types**: Bar, pie, line charts for basic data visualization
- **Responsive Design**: Built-in responsive design and mobile optimization
- **Interactive Features**: Hover tooltips, legend, basic interactions
- **Static Data**: Pre-loaded data from analysis results embedded in HTML
- **Self-Contained**: Chart.js library embedded inline (~200KB)

#### **Navigation in Static HTML**

Since the dashboard generates a single static HTML file, navigation is implemented through:

- **Tabbed Interfaces**: Channel and analysis type tabs for data organization
- **Expandable Sections**: JavaScript-controlled show/hide sections for detailed views
- **Progressive Disclosure**: Summary cards that expand to show detailed charts and data
- **Basic Interactions**: Chart.js hover tooltips and legend interactions

#### **Export Functionality in Static HTML**

Since the dashboard generates a single static HTML file, export functionality is implemented through:

- **Print Functionality**: Browser's built-in print capability for entire dashboard
- **Save as Image**: Browser's built-in screenshot capabilities for individual charts
- **PDF Generation**: Browser's "Save as PDF" functionality for complete dashboard
- **Share Links**: Direct file sharing of the generated HTML file

## Error Handling

### Error Handling Strategy

#### **Data Processing Errors**
- **File Not Found**: Skip missing analysis result files, log warnings, continue processing
- **Invalid JSON**: Skip corrupted files, log errors, continue with available data
- **Data Validation**: Check required fields, provide clear error messages
- **Empty Data**: Handle cases where no analysis data is available

#### **Output Generation Errors**
- **Directory Creation**: Create output directory if it doesn't exist, handle permission errors
- **File Writing**: Handle file write permission and disk space issues gracefully
- **Template Processing**: Handle template syntax errors with fallback to basic HTML
- **Chart Generation**: Handle chart configuration errors, skip problematic charts

### Performance Considerations

#### **Data Size Limits**
- **Maximum Data Points**: Limit charts to 10,000 data points (from `config.py` `DASHBOARD_MAX_DATA_POINTS`)
- **File Size**: Target HTML file size under 5MB for reasonable loading times
- **Memory Usage**: Process data in chunks for large datasets

## Testing Strategy

### Testing Approach

#### **Unit Testing**
- **Data Processing**: Test JSON parsing and data transformation
- **Template Rendering**: Test HTML template generation
- **Chart Configuration**: Test Chart.js chart configuration generation
- **File Operations**: Test input/output file handling

#### **Integration Testing**
- **End-to-End Testing**: Test complete HTML generation workflow
- **Data Integration**: Test analysis result file processing
- **Output Validation**: Test generated HTML file structure and content

### Testing Tools

#### **Python Testing**
- **Pytest**: Python testing framework
- **Pathlib Testing**: File system operation testing
- **JSON Validation**: Data structure validation testing

## Conclusion

The `dashboard` command design provides a simple and effective data visualization solution for Telegram channel analysis results. By generating a single self-contained HTML file with interactive Chart.js visualizations, it addresses the need for visual data exploration without requiring server infrastructure.

The simplified design emphasizes:
- **Single HTML File**: One comprehensive dashboard file instead of multiple files
- **Lightweight Technology**: Chart.js (~200KB) instead of Plotly (3MB) for faster loading
- **Analysis Data Integration**: Direct processing of analysis command output
- **Basic Visualizations**: Bar, pie, and line charts for essential data exploration
- **Simple Architecture**: Minimal dependencies and straightforward Python implementation
- **Portable Output**: Single HTML file that can be shared and viewed offline
- **Consistent CLI**: Follows established project patterns with 4 focused options
- **Configuration Management**: All constants centralized in `config.py`
- **Error Resilience**: Graceful handling of missing data and processing errors

The simplified approach with a single HTML file, lightweight charting library, and focused CLI ensures maximum usability while providing the essential visualization capabilities needed for exploring Telegram channel analysis results.
