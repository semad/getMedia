# Dashboard Implementation Specification

## Overview

This document provides detailed implementation specifications for the Dashboard command, based on the Dashboard General Design, HTML prototypes, and existing configuration structure. The implementation will generate a multi-page HTML dashboard website from analysis command output.

## Key Requirements

**Single Module Architecture:** Use `dashboard_processor.py` containing all dashboard functionality (data processing, template rendering, file generation, single page creation).

**Single Page Module:** Generate self-contained HTML files with embedded CSS, JavaScript, and data for email sharing, embedding, offline viewing, and quick reports. Target files under 2MB.

## Implementation Architecture

**Core Component:** `modules/dashboard_processor.py` - Single module containing all dashboard functionality including data processing, template rendering, file generation, single page creation, Chart.js integration, and error handling.

**Configuration:** Extend `config.py` with dashboard-specific constants for file paths, chart settings, and single page module configuration.

## Configuration Extensions

Add the following constants to `config.py` (reusing existing `ANALYSIS_BASE`, `DASHBOARDS_DIR`, `DEFAULT_GA_MEASUREMENT_ID`):

**Core Configuration:**

- `DASHBOARD_INPUT_DIR`, `DASHBOARD_OUTPUT_DIR`, `DASHBOARD_INDEX_FILENAME`, `DASHBOARD_CSS_FILENAME`, `DASHBOARD_JS_FILENAME`, `DASHBOARD_DATA_FILENAME`

**Paths and UI:**

- `DASHBOARD_CSS_PATH`, `DASHBOARD_JS_PATH`, `DASHBOARD_HTML_TITLE`, `DASHBOARD_HTML_CHARSET`, `DASHBOARD_HTML_VIEWPORT`

**Data Processing:**

- `DASHBOARD_DEFAULT_CHANNELS`, `DASHBOARD_SUPPORTED_ANALYSIS_TYPES`, `DASHBOARD_SUPPORTED_SOURCE_TYPES`, `DASHBOARD_MAX_CHANNEL_NAME_LENGTH`

**Charts and Analytics:**

- `DASHBOARD_CHART_WIDTH`, `DASHBOARD_CHART_HEIGHT`, `DASHBOARD_MAX_DATA_POINTS`, `DASHBOARD_CHARTJS_CDN_URL`, `DASHBOARD_GA_MEASUREMENT_ID`, `DASHBOARD_GA_ENABLED`

**Single Page Module:**

- `DASHBOARD_SINGLE_PAGE_ENABLED`, `DASHBOARD_SINGLE_PAGE_FILENAME`, `DASHBOARD_SINGLE_PAGE_MAX_SIZE_MB`

## File System Requirements

### Directory Structure

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
  - `static/` - Static files
    - `css/dashboard.css` - Generated styles
    - `js/dashboard.js` - Generated JavaScript
  - `data/dashboard-data.json` - Generated analysis data

### File Permissions

- **Read Access**: Required for analysis input files
- **Write Access**: Required for dashboard output directory
- **Execute Access**: Required for creating subdirectories

## Input Data Requirements

### Analysis Command Output

The dashboard requires analysis command output in the following format:

**Input Directory Structure:**

- `reports/analysis/file_messages/channels/channel1/`
  - `filename_analysis.json`
  - `filesize_analysis.json`
  - `message_analysis.json`
  - `analysis_summary.json`
- `reports/analysis/db_messages/channels/channel2/`
- `reports/analysis/diff_messages/channels/channel3/`

### JSON File Format

Each analysis file must contain:

**Required Structure:**

- `report_type`: Type of analysis (filename_analysis, filesize_analysis, message_analysis, analysis_summary)
- `generated_at`: ISO timestamp of analysis generation
- `analysis_version`: Version of analysis format
- `data`: Analysis-specific data object

### Single Page Module Data Validation

- **Data Size Limits**: Validate embedded data doesn't exceed memory limits
- **JSON Serialization**: Ensure all data can be serialized to JSON for embedding
- **Required Fields Validation**: Check for required fields before embedding
- **Data Sanitization**: Sanitize data to prevent XSS in embedded content
- **Channel Name Validation**: Validate channel names are safe for file system use
- **Metadata Validation**: Ensure metadata contains required generation information
- **Analysis Data Validation**: Validate analysis data structure before embedding

## Development Environment Setup

### Python Dependencies

**Core dependencies:**

- jinja2>=3.1.0
- click>=8.0.0
- pathlib2>=2.3.0 (for Python < 3.4 compatibility)

**Development dependencies:**

- pytest>=7.0.0
- pytest-cov>=4.0.0
- black>=22.0.0
- flake8>=5.0.0

### Required Python Version

- **Minimum**: Python 3.8+
- **Recommended**: Python 3.9+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11

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

Add dashboard command to `main.py` with click options for input-dir, output-dir, channels, single-page, and verbose flags. Import DashboardProcessor and handle exceptions.

### Step 3: Complete Module Implementation

The `dashboard_processor.py` module should include:

**Data Processing Methods:**

- `_discover_analysis_files()`, `_load_analysis_data()`, `_aggregate_channel_data()`
- `_extract_channel_name()`, `_sanitize_channel_name()`, `_update_summary_metrics()`

**Template Rendering Methods:**

- `_render_index_page()`, `_render_channel_page()`, `_render_shared_css()`, `_render_shared_js()`

**File Generation Methods:**

- `_generate_shared_files()`, `_generate_html_pages()`, `_generate_single_pages()`, `_create_empty_dashboard_data()`

### Step 4: HTML Templates

Create `templates/dashboard/index.html`, `channel.html`, and `single.html` with Jinja2 syntax, Google Analytics integration, responsive design, and Chart.js support. Adapt existing `dashboard_index.html` and `dashboard_channel.html` templates.

Generate self-contained HTML files with embedded CSS, JavaScript, and data. Include methods for CSS/JS embedding, data serialization, Chart.js integration, and file size validation.

## Google Analytics Integration

Implement Google Analytics 4 tracking for page views, chart interactions, and export actions. Include privacy controls with configurable enable/disable flag. Track anonymous usage patterns only.

## Error Handling & Performance

**Error Handling:**

- Skip missing files with warnings, handle corrupted JSON gracefully, provide fallbacks for missing data
- Create output directories if missing, handle permission errors, continue processing on individual file failures

**Performance Requirements:**

- Limit charts to 10,000 data points, target HTML files under 5MB, process large datasets in chunks
- Minify JSON data, use efficient chart rendering, implement lazy loading, cache processed data

## Browser Compatibility

### Supported Browsers

- **Chrome**: Version 90+
- **Firefox**: Version 88+
- **Safari**: Version 14+
- **Edge**: Version 90+

### Required Features

- **CSS Grid**: For responsive layouts
- **ES6 JavaScript**: For modern JavaScript features
- **Fetch API**: For data loading

## Security Requirements

### Data Privacy

- **No Personal Data**: Dashboard does not collect or store personal information
- **Anonymous Analytics**: Google Analytics tracking is anonymous
- **Local Processing**: All data processing happens locally

### File Security

- **Input Validation**: Validate all JSON input files
- **Path Sanitization**: Sanitize file paths to prevent directory traversal
- **Safe Templates**: Use Jinja2 autoescape for XSS prevention

## Testing Strategy

**Unit Tests:** Cover DashboardProcessor initialization, data processing, template rendering, error handling, and single page generation.

**Integration Tests:** End-to-end dashboard generation workflow, file output verification, error scenarios, and performance testing with large datasets.

**Test Data:** Sample analysis JSON files, various channel configurations, and edge cases (empty data, missing files, corrupted data).

## Deployment and Usage

**Installation:** Add dashboard modules, update `config.py` with constants, add CLI command to `main.py`, install dependencies.

**Usage Examples:**

- `python main.py dashboard` - Default settings
- `python main.py dashboard -c "books,@SherwinVakiliLibrary"` - Specific channels
- `python main.py dashboard -i /path/to/analysis -o /path/to/output` - Custom directories
- `python main.py dashboard -v` - Verbose logging

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
