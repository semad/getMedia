# Dashboard Implementation Specification

## Overview

This document provides detailed implementation specifications for the Dashboard command, based on the Dashboard General Design, HTML prototypes, and existing configuration structure. The implementation will generate a multi-page HTML dashboard website from analysis command output.

## Implementation Requirements

### Development Environment Setup

#### **Python Dependencies**

**Core dependencies:**

- jinja2>=3.1.0
- click>=8.0.0
- pathlib2>=2.3.0 (for Python < 3.4 compatibility)

**Development dependencies:**

- pytest>=7.0.0
- pytest-cov>=4.0.0
- black>=22.0.0
- flake8>=5.0.0

#### **Required Python Version**

- **Minimum**: Python 3.8+
- **Recommended**: Python 3.9+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11

### File System Requirements

#### **Directory Structure**

The implementation requires the following directory structure to exist or be created:

**Source Structure:**

- `getMedia/modules/dashboard_processor.py` - Complete dashboard processor
- `getMedia/templates/dashboard/` - HTML templates directory
  - `base.html` - Base template
  - `index.html` - Index page template
  - `channel.html` - Channel page template
  - `single.html` - Single page template
  - `components/` - Template components
    - `header.html` - Header component
    - `metrics.html` - Metrics cards
    - `charts.html` - Chart containers
    - `footer.html` - Footer component

**Input Structure:**

- `getMedia/reports/analysis/` - Input data directory

**Output Structure:**

- `getMedia/reports/dashboards/html/` - Output directory
  - `index.html` - Generated main dashboard
  - `{channel}.html` - Generated channel pages
  - `dashboard-standalone.html` - Generated single page dashboards
  - `dashboard/` - Static files
    - `css/dashboard.css` - Generated styles
    - `js/dashboard.js` - Generated JavaScript
  - `data/dashboard-data.json` - Generated analysis data

#### **File Permissions**

- **Read Access**: Required for analysis input files
- **Write Access**: Required for dashboard output directory
- **Execute Access**: Required for creating subdirectories

### Input Data Requirements

#### **Analysis Command Output**

The dashboard requires analysis command output in the following format:

**Input Directory Structure:**

- `reports/analysis/file_messages/channels/channel1/`
  - `filename_analysis.json`
  - `filesize_analysis.json`
  - `message_analysis.json`
  - `analysis_summary.json`
- `reports/analysis/db_messages/channels/channel2/`
- `reports/analysis/diff_messages/channels/channel3/`

#### **JSON File Format**

Each analysis file must contain:

**Required Structure:**

- `report_type`: Type of analysis (filename_analysis, filesize_analysis, message_analysis, analysis_summary)
- `generated_at`: ISO timestamp of analysis generation
- `analysis_version`: Version of analysis format
- `data`: Analysis-specific data object

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

**Dashboard Configuration:**

- `DASHBOARD_INPUT_DIR` = ANALYSIS_BASE (already exists)
- `DASHBOARD_OUTPUT_DIR` = DASHBOARDS_DIR (already exists)

**File Naming:**

- `DASHBOARD_INDEX_FILENAME` = "index.html"
- `DASHBOARD_CSS_FILENAME` = "dashboard.css"
- `DASHBOARD_JS_FILENAME` = "dashboard.js"
- `DASHBOARD_DATA_FILENAME` = "dashboard-data.json"

**Static File Paths:**

- `DASHBOARD_CSS_PATH` = "dashboard/css"
- `DASHBOARD_JS_PATH` = "dashboard/js"

**HTML Generation:**

- `DASHBOARD_HTML_TITLE` = "Telegram Channel Analysis Dashboard"
- `DASHBOARD_HTML_CHARSET` = "UTF-8"
- `DASHBOARD_HTML_VIEWPORT` = "width=device-width, initial-scale=1.0"

**Data Processing:**

- `DASHBOARD_DEFAULT_CHANNELS` = "all"
- `DASHBOARD_SUPPORTED_ANALYSIS_TYPES` = ["filename", "filesize", "message"]
- `DASHBOARD_SUPPORTED_SOURCE_TYPES` = ["file_messages", "db_messages", "diff_messages"]
- `DASHBOARD_MAX_CHANNEL_NAME_LENGTH` = 50

**Chart Configuration:**

- `DASHBOARD_CHART_WIDTH` = 800
- `DASHBOARD_CHART_HEIGHT` = 600
- `DASHBOARD_MAX_DATA_POINTS` = 10000
- `DASHBOARD_CHARTJS_CDN_URL` = <https://cdn.jsdelivr.net/npm/chart.js>

**Google Analytics Configuration:**

- `DASHBOARD_GA_MEASUREMENT_ID` = DEFAULT_GA_MEASUREMENT_ID (from existing config)
- `DASHBOARD_GA_ENABLED` = True

**Single Page Module Configuration:**

- `DASHBOARD_SINGLE_PAGE_ENABLED` = True
- `DASHBOARD_SINGLE_PAGE_FILENAME` = "dashboard-standalone.html"
- `DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB` = 2

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

**Required Imports:**

- json, logging, datetime, pathlib, typing
- jinja2 (Environment, FileSystemLoader)
- config (all dashboard constants)

**DashboardProcessor Class Structure:**

- `__init__()` - Initialize with input/output directories, channels, single_page flag, verbose mode
- `_setup_logger()` - Configure logging
- `_setup_templates()` - Setup Jinja2 template environment
- `_parse_channels()` - Parse comma-separated channel list
- `process()` - Main processing method that generates complete dashboard

**Core Functionality:**

- Data processing and aggregation
- Template rendering
- File generation
- Single page module creation
- Error handling

### Step 2: CLI Integration

#### Add to `main.py`

**CLI Command Structure:**

- `@cli.command()` decorator for dashboard command
- Click options for input-dir, output-dir, channels, single-page, verbose
- Import DashboardProcessor from modules.dashboard_processor
- Initialize processor with provided parameters
- Call processor.process() method
- Handle ImportError and general exceptions with appropriate error messages

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

**Required Template Files:**

- `templates/dashboard/index.html` - Main dashboard page
- `templates/dashboard/channel.html` - Individual channel page  
- `templates/dashboard/single.html` - Single page template (optional)

**Template Features:**

- Jinja2 template syntax with variables and loops
- Google Analytics integration with conditional loading
- Responsive design with CSS Grid and Flexbox
- Chart.js integration for data visualization
- Navigation and action buttons
- Metric cards and channel grids

**Existing Templates to Adapt:**

- `templates/dashboard_index.html` → `templates/dashboard/index.html`
- `templates/dashboard_channel.html` → `templates/dashboard/channel.html`

### Step 4: Single Page Module

**Single Page Module Features:**

- Self-contained HTML files with embedded CSS, JavaScript, and data
- No external dependencies for offline viewing
- File size validation and optimization
- Google Analytics integration
- Chart.js CDN or embedded version support
- Channel-specific and full dashboard views

**Key Methods:**

- `generate_single_page()` - Main generation method
- `_get_embedded_css()` - Extract CSS for embedding
- `_get_embedded_js()` - Extract JavaScript for embedding
- `_get_embedded_data()` - Convert data to JavaScript variables
- `_get_chartjs_script()` - Get Chart.js script tag
- `_render_single_page_template()` - Render complete HTML with embedded resources
- `_get_single_page_filename()` - Generate appropriate filename

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

**Test Coverage Areas:**

- DashboardProcessor initialization and configuration
- Data processing and file discovery
- Template rendering and file generation
- Error handling and edge cases
- Single page module generation

**Key Test Functions:**

- `test_dashboard_generator_initialization()` - Verify proper initialization
- `test_data_processor_file_discovery()` - Test file discovery functionality
- `test_template_rendering()` - Test template rendering with various data
- `test_single_page_generation()` - Test single page module creation

### Integration Tests

**End-to-End Testing:**

- Complete dashboard generation workflow
- File output verification
- Single page module generation
- Error handling scenarios
- Performance testing with large datasets

**Test Data Requirements:**

- Sample analysis JSON files
- Various channel configurations
- Edge cases (empty data, missing files, corrupted data)

## Deployment and Usage

### Installation

1. Add dashboard modules to the project
2. Update `config.py` with dashboard constants
3. Add CLI command to `main.py`
4. Install required dependencies

### Usage Examples

**Basic Usage:**

- `python main.py dashboard` - Generate dashboard with default settings
- `python main.py dashboard -c "books,@SherwinVakiliLibrary"` - Generate for specific channels
- `python main.py dashboard -i /path/to/analysis -o /path/to/output` - Custom directories
- `python main.py dashboard -v` - Verbose logging

**Single Page Module:**

- `python main.py dashboard -s` - Generate single-page standalone dashboards
- `python main.py dashboard -s -c "books,@SherwinVakiliLibrary"` - Single-page for specific channels
- `python main.py dashboard -s -v` - Both multi-page and single-page dashboards

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
