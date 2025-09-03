# Analysis Command Design Specification

## Overview

The `analysis` command is an advanced intermediate data analysis tool for Telegram channel data, providing comprehensive data analysis capabilities across multiple sources. It supports file-based data, API endpoints, and diff comparison with pattern recognition, statistical analysis, and detailed reporting.

## Design Goals

1. **Advanced Intermediate Data Analysis Tool**: Comprehensive data analysis with pattern recognition, statistical analysis, and detailed insights
2. **Single Module Architecture**: All analysis functionality consolidated into one module for simplicity and maintainability
3. **Flexible Data Sources**: Support for file-based data, API endpoints, and diff analysis
4. **Advanced Intermediate Analytics**: Pattern recognition, language detection, engagement analysis, and statistical metrics
5. **Performance Optimization**: Efficient data processing using pandas DataFrames with vectorized operations
6. **Detailed Output**: JSON reports with comprehensive analysis results and metadata
7. **Extensibility**: Easy to add new analysis types and advanced features in future versions
8. **Error Handling**: Comprehensive error handling and validation throughout the pipeline

## Command Line Interface

### Basic Usage
```bash
python main.py analysis [OPTIONS]
```

### CLI Options

#### **Core Options**
- `--channels, -c TEXT`: Comma-separated list of channel usernames to analyze
- `--analysis-types, -t TEXT`: Comma-separated list of analysis types: filename,filesize,message (default: all)
- `--output-dir, -o PATH`: Output directory for analysis results (default: analysis_output_<timestamp>)
- `--chunk-size INTEGER`: Chunk size for processing large datasets (default: 10000)
- `--verbose, -v`: Enable verbose logging output
- `--help, -h`: Show help message and exit

#### **Data Source Control Options**
- `--enable-file-source / --no-file-source`: Enable/disable file-based data sources (default: enabled)
- `--enable-api-source / --no-api-source`: Enable/disable API-based data sources (default: enabled)
- `--enable-diff-analysis / --no-diff-analysis`: Enable/disable diff analysis between file and API sources (default: enabled)

#### **API Configuration Options**
- `--api-base-url TEXT`: Base URL for API endpoints (default: http://localhost:8000)
- `--api-timeout INTEGER`: API request timeout in seconds (default: 30)
- `--items-per-page INTEGER`: Number of items per API page (default: 100)

#### **Performance and Reliability Options**
- `--memory-limit INTEGER`: Memory limit for processing in MB (default: 100000)
- `--retry-attempts INTEGER`: Number of retry attempts for failed operations (default: 3)
- `--retry-delay FLOAT`: Delay between retry attempts in seconds (default: 1.0)

### Usage Examples

#### **Basic Analysis**
```bash
# Analyze all data with default settings
python main.py analysis

# Analyze specific channels
python main.py analysis --channels "@channel1,@channel2"

# Run specific analysis types
python main.py analysis --analysis-types filename,filesize

# Specify output directory
python main.py analysis --output-dir ./results

# Use smaller chunks for large datasets
python main.py analysis --chunk-size 5000

# Enable verbose logging
python main.py analysis --verbose
```

#### **Data Source Control**
```bash
# Use only API sources
python main.py analysis --no-file-source

# Use only file sources
python main.py analysis --no-api-source

# Disable diff analysis
python main.py analysis --no-diff-analysis

# Combine file and API sources (diff analysis)
python main.py analysis --enable-file-source --enable-api-source
```

#### **API Configuration**
```bash
# Custom API endpoint
python main.py analysis --api-base-url http://api.example.com

# Custom API timeout
python main.py analysis --api-timeout 60

# Custom items per page
python main.py analysis --items-per-page 200
```

#### **Performance Tuning**
```bash
# Custom memory limit
python main.py analysis --memory-limit 200000

# Custom retry settings
python main.py analysis --retry-attempts 5 --retry-delay 2.0

# Combined performance settings
python main.py analysis --chunk-size 5000 --memory-limit 150000 --retry-attempts 3
```

## Requirements

### Functional Requirements

#### **REQ-001: Filename Analysis**
The analysis command MUST provide comprehensive filename analysis capabilities including:

- **REQ-001.1**: **Duplicate Filename Detection**
  - Identify files with identical names (exact matches)
  - Count the number of files sharing each filename
  - Generate a list of most common filenames (top 10)
  - Calculate the ratio of unique filenames vs total files
  - Provide actionable data for duplicate file cleanup

- **REQ-001.2**: **Filename Pattern Analysis**
  - Analyze filename length distribution (min, max, mean, median)
  - Identify common filename patterns and extensions
  - Detect files with special characters or unusual naming conventions
  - Categorize files by extension type and frequency
  - Generate statistics on naming convention compliance

- **REQ-001.3**: **Filename Quality Assessment**
  - Identify files with problematic naming patterns
  - Flag files with spaces, special characters, or non-standard formats
  - Provide recommendations for filename standardization
  - Generate quality scores for filename consistency

#### **REQ-002: Filesize Analysis**
The analysis command MUST provide comprehensive filesize analysis capabilities including:

- **REQ-002.1**: **Duplicate Filesize Detection**
  - Identify files with identical sizes (exact byte matches)
  - Count the number of files sharing each filesize
  - Generate a list of most common filesizes (top 10)
  - Calculate the ratio of unique filesizes vs total files
  - Provide potential duplicate file identification based on size

- **REQ-002.2**: **Filesize Distribution Analysis**
  - Create meaningful size bins (0-1MB, 1-5MB, 5-10MB, 10MB+)
  - Analyze filesize frequency distribution
  - Identify size clusters and patterns
  - Generate storage optimization recommendations

#### **REQ-003: Language Analysis**
The analysis command MUST provide comprehensive language analysis capabilities including:

- **REQ-003.1**: **Language Detection**
  - Detect primary language of message content
  - Identify language distribution across all messages
  - Handle unknown or mixed language content gracefully
  - Provide fallback mechanisms for detection failures

- **REQ-003.2**: **Language Statistics**
  - Calculate percentage distribution of detected languages
  - Identify most common languages (top 5)
  - Generate language diversity metrics
  - Provide language-specific content analysis

#### **REQ-004: Data Source Integration**
The analysis command MUST support multiple data sources:

- **REQ-004.1**: **File Source Support**
  - Read from combined JSON files in `reports/collections/` directory
  - Support files matching `tg_*_combined.json` pattern
  - Handle both small and large files efficiently
  - Provide data quality assessment for file sources

- **REQ-004.2**: **API Source Support**
  - Fetch data from REST API endpoints
  - Support pagination for large datasets
  - Handle API timeouts and connection errors gracefully
  - Provide real-time data availability checks

- **REQ-004.3**: **Dual Source Comparison**
  - Compare data between file and API sources
  - Calculate sync percentages and status
  - Identify missing records in each source
  - Generate discrepancy reports

### Non-Functional Requirements

#### **REQ-005: Performance Requirements**
- **REQ-005.1**: Process datasets up to 100,000 records within 60 seconds
- **REQ-005.2**: Use memory-efficient pandas operations for large datasets
- **REQ-005.3**: Support chunked processing for files exceeding available memory
- **REQ-005.4**: Provide progress indicators for operations longer than 30 seconds

#### **REQ-006: Output Requirements**
- **REQ-006.1**: Generate structured JSON output with comprehensive analysis results
- **REQ-006.2**: Include detailed metadata and processing statistics
- **REQ-006.3**: Provide actionable insights for media library optimization
- **REQ-006.4**: Support both individual channel and summary reports
- **REQ-006.5**: **Individual Channel Directory Structure**: Create separate folders for each channel when analyzing all channels
  - Each channel gets its own directory: `reports/analysis/file_messages/{channel_name}/`
  - Single channel analysis: `reports/analysis/file_messages/books/`
  - Multiple channel analysis: `reports/analysis/file_messages/combined_X_channels/`
  - All channels analysis: Individual folders for each discovered channel
- **REQ-006.6**: **Channel-Specific File Naming**: Use channel names in output file names
  - Comprehensive report: `{channel_name}_analysis.json`
  - Individual reports: `filename_analysis.json`, `filesize_analysis.json`, `message_analysis.json`
  - Summary report: `analysis_summary.json`
- **REQ-006.7**: **Config.py Compliance**: All output paths must follow patterns defined in `config.py`
  - Use `FILE_MESSAGES_DIR` for base directory
  - Use `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN` for file naming
  - Ensure directory structure matches configuration requirements
- **REQ-006.8**: **Diff Analysis Output Structure**: Diff analysis results must be output to separate directory structure
  - Diff analysis output: `reports/analysis/diff_messages/{channel_name}/`
  - File-based analysis output: `reports/analysis/file_messages/{channel_name}/`
  - API-based analysis output: `reports/analysis/db_messages/{channel_name}/`
  - Use `DIFF_MESSAGES_DIR` for diff analysis base directory
  - Maintain same file naming patterns within each directory type

#### **REQ-007: Error Handling Requirements**
- **REQ-007.1**: Continue processing with partial data when possible
- **REQ-007.2**: Provide clear, actionable error messages with specific error codes
- **REQ-007.3**: Log all errors with timestamps and context
- **REQ-007.4**: Generate error summaries in output metadata

## Design Constraints

### Technology Constraints
- **Pandas Library**: All data processing and analysis must use the Pandas library for:
  - Data loading from JSON files and API responses
  - Data manipulation and transformation
  - Statistical calculations and aggregations
  - Data filtering and grouping operations
  - Performance optimization through vectorized operations
- **Python 3.8+**: Minimum Python version requirement for Pandas compatibility
- **Memory Efficiency**: Large dataset handling through Pandas chunking and memory management

### Dependencies
The analysis module requires the following Python packages:

#### **Core Dependencies**
- **pandas**: >=1.5.0 (data manipulation and analysis)
- **numpy**: >=1.21.0 (numerical operations)
- **aiohttp**: >=3.8.0 (async HTTP client for API calls)
- **asyncio**: built-in (async programming support)
- **json**: built-in (JSON processing)
- **pathlib**: built-in (file path handling)
- **logging**: built-in (logging functionality)
- **datetime**: built-in (date/time operations)

#### **Required Dependencies**
- **langdetect**: >=1.0.9 (language detection)
- **emoji**: >=2.0.0 (emoji analysis)

#### **Installation Requirements**
```bash
pip install pandas>=1.5.0 numpy>=1.21.0 aiohttp>=3.8.0
pip install langdetect>=1.0.9 emoji>=2.0.0
```

### Performance Constraints
- **DataFrame Operations**: Leverage Pandas' optimized C-based operations for large datasets
- **Chunked Processing**: Use Pandas chunking for files exceeding available memory
- **Vectorized Operations**: Prefer Pandas vectorized operations over Python loops

## Architecture Overview

### High-Level Components

```
┌─────────────────┐    ┌─────────────────────────────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │         Single Analysis Module         │    │   Output Layer  │
│                 │    │                                         │    │                 │
│ - Command       │───▶│ ┌─────────────┐ ┌─────────────────────┐ │───▶│ - JSON Format   │
│ - Options       │    │ │Data Loading │ │   Analysis Core     │ │    │ - File Writing  │
│ - Validation    │    │ │- File       │ │ - Message           │ │    │ - Summary Gen   │
└─────────────────┘    │ │- API        │ │ - Media             │ │    └─────────────────┘
                       │ │- Dual       │ │ - Diff              │ │
                       │ └─────────────┘ └─────────────────────┘ │
                       │ ┌─────────────┐ ┌─────────────────────┐ │
                       │ │Orchestrator │ │   Output Formatter  │ │
                       │ │- Pipeline   │ │ - JSON Generation   │ │
                       │ │- Validation │ │ - File Management   │ │
                       │ │- Progress   │ └─────────────────────┘ │
                       │ └─────────────┘                         │
                       └─────────────────────────────────────────┘
```

### Data Flow Architecture

1. **Input**: CLI arguments and configuration
2. **Data Discovery**: Determine available data from API endpoints or combined collections directory
3. **Data Loading**: Source-specific data loaders within the single module (file, API, or dual-source)
4. **Validation**: Data quality checks and validation
5. **Analysis**: Execution of selected analysis types
6. **Output**: JSON formatting and file generation
7. **Metadata**: Generation of analysis metadata and summaries

## CLI Interface

### Command Structure

```bash
python main.py analysis [options]
```

### Analysis Type
The analysis command runs **advanced intermediate analysis** providing comprehensive data analysis capabilities:

#### **Core Analysis Types (Always Executed)**
- **Message Analysis**: Comprehensive content statistics and pattern recognition
- **Media Analysis**: Detailed file type distribution and storage metrics

#### **Comparison Analysis (Enabled by default, can be disabled with --no-diff)**
- **Diff Analysis**: Detailed record count comparison and sync status

**Note**: This is an advanced intermediate analysis tool providing comprehensive metrics and insights. Additional advanced features (sentiment analysis, topic modeling, network analysis, etc.) will be added in future versions.

### Command Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--no-file` | | flag | `False` | Disable file source (combined collections) |
| `--no-api` | | flag | `False` | Disable API source (REST endpoints) |
| `--no-diff` | | flag | `False` | Disable diff analysis (file vs API comparison) |
| `--channels` | `-c` | string | `all` | Comma-separated channel list or `all` |
| `--verbose` | `-v` | flag | `False` | Enable verbose logging |
| `--help` | `-h` | flag | | Show help message |

**Note**: All sources (file, API, diff) are enabled by default, but can be individually disabled with their respective `--no-*` flags.

### Flag Validation Rules

The CLI enforces these validation rules:

1. **Default behavior**: All sources (file, API, diff) are enabled by default
2. **Individual source control**: Each source can be disabled with its respective `--no-*` flag
3. **Diff validation**: Diff analysis requires both file and API sources to be available
4. **Minimum source requirement**: At least one source (file or API) must be enabled

### Usage Examples

```bash
# Default behavior (all sources and diff analysis)
python main.py analysis

# Analysis with specific channels
python main.py analysis --channels @SherwinVakiliLibrary

# Analysis with verbose logging
python main.py analysis --verbose

# Analysis with specific channels and verbose logging
python main.py analysis --channels @SherwinVakiliLibrary --verbose

# Analysis without file source (API only)
python main.py analysis --no-file

# Analysis without API source (file only)
python main.py analysis --no-api

# Analysis without diff comparison (file and API separately)
python main.py analysis --no-diff

# Analysis with only file source (no API, no diff)
python main.py analysis --no-api --no-diff

# Analysis with only API source (no file, no diff)
python main.py analysis --no-file --no-diff

# Analysis with specific channels and no diff
python main.py analysis --channels @SherwinVakiliLibrary --no-diff
```

## Data Discovery and Sources

### Data Discovery Phase
The analysis command first determines what data is available using concrete metrics:

- **API Discovery**: When API source is enabled (not disabled with `--no-api`), queries available endpoints to find:
  - Available channels (`/api/v1/telegram/channels`) - must return HTTP 200
  - Message counts and data ranges from `/api/v1/telegram/messages` endpoint
  - API health check - response time <30 seconds, no HTTP errors
  - Comprehensive data availability check

- **File Discovery**: When file source is enabled (not disabled with `--no-file`), scans the combined collections directory to find:
  - Available combined JSON files matching `*_combined.json` pattern
  - Channel names extracted from file metadata
  - Message counts from file metadata (any files with messages)
  - File sizes and modification dates (no age restrictions)

- **Dual Source Discovery**: When diff analysis is enabled (not disabled with `--no-diff`), performs both API and file discovery to compare:
  - Channel overlap between sources
  - Message count differences
  - Data freshness comparison
  - Quality score comparison

### File Source (`--no-file`)
- Reads from combined JSON files in `COLLECTIONS_DIR`
- Uses pandas for efficient JSON processing
- Supports both small and large files with appropriate strategies
- Can be disabled with `--no-file` flag

### API Source (`--no-api`)
- Fetches data from REST API endpoints
- Converts API responses to pandas DataFrames
- Supports async operations for better performance
- Can be disabled with `--no-api` flag

### Diff Source (`--no-diff`)
- **Requires both file and API sources** to be available (not disabled with `--no-file` or `--no-api`)
- Loads data from both file and API sources
- Adds source identifiers to distinguish data origins
- Enables comparison and validation between sources
- Generates comparison reports showing differences and sync status
- Can be disabled with `--no-diff` flag

## Analysis Types

### Message Analysis (Advanced Intermediate Scope)

#### **Comprehensive Content Statistics**
- **Message Count**: Total messages, messages with text, empty messages
- **Text Length Analysis**: 
  - Character count (min, max, mean, median)
  - Word count (min, max, mean, median)
  - Message length categories (short: <50 chars, medium: 50-200, long: >200)
- **Content Type Distribution**: Text-only, media-only, text+media, empty messages

#### **Pattern Recognition**
- **Hashtag Analysis**: 
  - Most common hashtags (top 10)
  - Total unique hashtags count
- **Mention Analysis**:
  - Most mentioned usernames (top 10)
  - Total unique mentions count
- **URL Analysis**:
  - Total URL count
  - Most common domains (top 5)
- **Emoji Analysis**:
  - Most used emojis (top 10)
  - Total unique emojis count

#### **Creator Analysis**
- **Active Contributors**: Top contributors by message count (top 10)
- **Creator Message Counts**: Detailed count per creator with engagement metrics

### Media Analysis (Advanced Intermediate Scope)

#### **File Size Analysis**
- **Storage Usage**: Total storage consumed, average file size
- **Size Distribution**: 
  - File size categories (small: <1MB, medium: 1-10MB, large: >10MB)
  - Size percentiles (50th, 75th, 90th)
- **Large File Identification**: Top 10 largest files

#### **Media Type Analysis**
- **File Format Distribution**: 
  - Document types (PDF, DOC, TXT, etc.)
  - Image types (JPG, PNG, GIF, etc.)
  - Video types (MP4, AVI, etc.)
  - Audio types (MP3, WAV, etc.)
- **MIME Type Analysis**: Most common MIME types (top 10)
- **Average File Sizes by Type**: Detailed size metrics per media type

#### **Filename Analysis**
- **Duplicate Filename Detection**:
  - Files with identical names (exact matches)
  - Count of files sharing each filename
  - Most common filenames (top 10)
  - Total unique filenames vs total files
- **Filename Pattern Analysis**:
  - Filename length distribution (min, max, mean, median)
  - Common filename patterns and extensions
  - Files with special characters or unusual naming conventions

#### **Filesize Analysis**
- **Duplicate Filesize Detection**:
  - Files with identical sizes (exact byte matches)
  - Count of files sharing each filesize
  - Most common filesizes (top 10)
  - Total unique filesizes vs total files
- **Filesize Distribution Analysis**:
  - Filesize frequency distribution
  - Size clusters and patterns
  - Potential duplicate file identification based on size

#### **Media Content Analysis**
- **Caption Analysis**: 
  - Caption presence (with/without captions)
  - Caption length (min, max, mean, median)
- **Media Engagement**: 
  - Views, forwards, replies by media type (if available)
  - Most popular media files (top 10)

### Diff Analysis (Advanced Intermediate Scope)
- **Source Comparison**: Detailed record count comparison between file and API sources
- **Sync Status**: Comprehensive sync percentage and status calculation:
  - **synced**: 100% match between sources (sync_percentage = 100)
  - **partially_synced**: 90-99% match between sources (sync_percentage >= 90)
  - **not_synced**: <90% match between sources (sync_percentage < 90)
- **Discrepancy Reporting**: Detailed count of missing records in each source:
  - **missing_in_api**: Records present in file but not in API (by message_id comparison)
  - **missing_in_file**: Records present in API but not in file (by message_id comparison)

## Data Processing Pipeline

1. **Data Loading**: Source selection, discovery, and loading strategy
2. **Data Validation**: Schema validation, quality assessment, and error reporting
3. **Analysis Execution**: Processor selection, data preparation, and result collection
4. **Output Generation**: JSON formatting, file organization, and metadata creation

### Individual Channel Report Generation

The analysis system implements a sophisticated report generation strategy that creates individual folders for each channel when analyzing all channels:

#### **Report Generation Logic**
1. **Channel Discovery**: Identify all unique channels from discovered data sources
2. **Channel Deduplication**: Remove duplicate channel entries (same channel from multiple sources)
3. **Individual Report Creation**: Generate separate reports for each unique channel
4. **Directory Structure Creation**: Create channel-specific directories following `config.py` patterns

#### **Channel Processing Flow**
```
Data Sources → Channel Discovery → Channel Deduplication → Individual Report Generation
     ↓              ↓                    ↓                        ↓
8 sources    → 5 unique channels  →  Remove duplicates  →  5 channel folders
```

#### **Report Generation Methods**
- **`_generate_individual_channel_reports`**: Main method for creating individual channel reports
- **`_get_output_paths`**: Generates channel-specific output paths following `config.py` patterns
- **Channel-specific directory creation**: Each channel gets its own folder with all analysis types

#### **Output File Types per Channel**
Each channel directory contains:
- **Comprehensive Report**: `{channel_name}_analysis.json` - Complete analysis results
- **Filename Analysis**: `filename_analysis.json` - Filename patterns and duplicates
- **Filesize Analysis**: `filesize_analysis.json` - File size distributions and duplicates
- **Message Analysis**: `message_analysis.json` - Message content and engagement analysis
- **Summary Report**: `analysis_summary.json` - High-level metrics and insights

## Output Structure

### Directory Organization

The analysis command generates output in a structured directory hierarchy that follows the `config.py` patterns and creates individual folders for each channel.

#### **Individual Channel Directory Structure**
When analyzing all channels (no specific channels provided), the system creates separate folders for each discovered channel:

```
reports/analysis/file_messages/
├── books/                           # @books channel
│   ├── books_analysis.json         # Comprehensive analysis report
│   ├── filename_analysis.json      # Filename analysis results
│   ├── filesize_analysis.json      # Filesize analysis results
│   ├── message_analysis.json       # Message analysis results
│   └── analysis_summary.json       # Summary report
├── books_magazine/                  # @books_magazine channel
│   ├── books_magazine_analysis.json
│   ├── filename_analysis.json
│   ├── filesize_analysis.json
│   ├── message_analysis.json
│   └── analysis_summary.json
├── Free/                           # @Free channel
│   ├── Free_analysis.json
│   ├── filename_analysis.json
│   ├── filesize_analysis.json
│   ├── message_analysis.json
│   └── analysis_summary.json
├── Free_Books_life/                # @Free_Books_life channel
│   ├── Free_Books_life_analysis.json
│   ├── filename_analysis.json
│   ├── filesize_analysis.json
│   ├── message_analysis.json
│   └── analysis_summary.json
└── SherwinVakiliLibrary/           # @SherwinVakiliLibrary channel
    ├── SherwinVakiliLibrary_analysis.json
    ├── filename_analysis.json
    ├── filesize_analysis.json
    ├── message_analysis.json
    └── analysis_summary.json
```

#### **Single Channel Analysis**
When analyzing a specific channel, files are created in that channel's directory:

```bash
# Command: python main.py analysis --channels @books
reports/analysis/file_messages/books/
├── books_analysis.json
├── filename_analysis.json
├── filesize_analysis.json
├── message_analysis.json
└── analysis_summary.json
```

#### **Multiple Channel Analysis**
When analyzing multiple specific channels, files are created in a combined directory:

```bash
# Command: python main.py analysis --channels @books,@Free
reports/analysis/file_messages/combined_2_channels/
├── combined_2_channels_analysis.json
├── filename_analysis.json
├── filesize_analysis.json
├── message_analysis.json
└── analysis_summary.json
```

#### **Diff Analysis Output Structure**
When diff analysis is enabled (both file and API sources present), output is generated in the `diff_messages` directory:

```bash
# Command: python main.py analysis --channels @books (with both file and API sources)
reports/analysis/diff_messages/books/
├── books_analysis.json              # Comprehensive diff analysis report
├── filename_analysis.json           # Filename analysis from combined sources
├── filesize_analysis.json           # Filesize analysis from combined sources
├── message_analysis.json            # Message analysis from combined sources
└── analysis_summary.json            # Summary of diff analysis results
```

#### **Data Source-Specific Output Structure**
The system automatically routes output based on data source types:

- **File-only sources**: `reports/analysis/file_messages/{channel_name}/`
- **API-only sources**: `reports/analysis/db_messages/{channel_name}/`
- **Mixed sources (diff analysis)**: `reports/analysis/diff_messages/{channel_name}/`

#### **Legacy Structure (Deprecated)**
The old structure is no longer used but documented for reference:

```
reports/analysis/
├── messages/
│   ├── {channel_name}_messages.json
│   └── messages_summary.json
├── media/
│   ├── {channel_name}_media.json
│   └── media_summary.json
└── diff/
    ├── {channel_name}_diff.json
    └── diff_summary.json
```

## Data Input Formats

### File Source Data Format
The analysis command reads from combined JSON files in the `reports/collections/` directory:

```json
[
  {
    "metadata": {
      "combined_at": "2025-09-01T17:33:24.499966",
      "channel": "@SherwinVakiliLibrary",
      "source_files": ["tg_SherwinVakiliLibrary_150199_150300.json"],
      "message_id_ranges": [[150199, 150300]],
      "total_messages": 200,
      "overall_range": "150199-150300",
      "data_format": "combined_collection",
      "source_directory": "raw",
      "new_messages_added": 100,
      "existing_messages": 100
    },
    "messages": [
      {
        "message_id": 150199,
        "channel_username": "@SherwinVakiliLibrary",
        "date": "2024-01-15T10:30:00Z",
        "text": "Message content here",
        "creator_username": "user123",
        "creator_first_name": "John",
        "creator_last_name": "Doe",
        "media_type": "document",
        "file_name": "document.pdf",
        "file_size": 1048576,
        "mime_type": "application/pdf",
        "caption": "Document caption",
        "views": 150,
        "forwards": 5,
        "replies": 2,
        "is_forwarded": false,
        "forwarded_from": null
      }
    ]
  }
]
```

### API Source Data Format
The analysis command fetches data from REST API endpoints:

**Channels Endpoint** (`/api/v1/telegram/channels`):
```json
[
  {
    "username": "@SherwinVakiliLibrary",
    "title": "Sherwin Vakili Library",
    "description": "Channel description",
    "member_count": 5000,
    "created_at": "2021-09-26T00:00:00Z"
  }
]
```

**Messages Endpoint** (`/api/v1/telegram/messages`):
```json
{
  "data": [
    {
      "id": 1,
      "message_id": 150199,
      "channel_username": "@SherwinVakiliLibrary",
      "date": "2024-01-15T10:30:00Z",
      "text": "Message content here",
      "media_type": "document",
      "file_name": "document.pdf",
      "file_size": 1048576,
      "mime_type": "application/pdf",
      "views": 150,
      "forwards": 5,
      "replies": 2,
      "creator_username": "user123",
      "creator_first_name": "John",
      "creator_last_name": "Doe"
    }
  ],
  "total_count": 400,
  "has_more": false,
  "page": 1,
  "items_per_page": 100
}
```

### Unified Data Structure
The analysis system normalizes data from both sources into a common schema:

#### **Common Message Schema**
```json
{
  "message_id": "unique_identifier",
  "channel_username": "channel_name",
  "date": "ISO_timestamp",
  "text": "message_content",
  "creator_username": "user_identifier",
  "creator_first_name": "first_name",
  "creator_last_name": "last_name",
  "media_type": "document|image|video|audio|none",
  "file_name": "filename.ext",
  "file_size": 1048576,
  "mime_type": "application/pdf",
  "caption": "media_caption",
  "views": 150,
  "forwards": 5,
  "replies": 2,
  "is_forwarded": false,
  "forwarded_from": null,
  "source": "file|api"
}
```

#### **Data Normalization Process**
1. **File Source**: Extract messages from nested `messages` array, add `source: "file"`
2. **API Source**: Flatten API response, add `source: "api"`
3. **Field Mapping**: Map different field names to common schema (see Field Mapping Table below)
4. **Type Conversion**: Ensure consistent data types (dates, numbers, booleans)
5. **Validation**: Verify required fields and data quality

#### **Field Mapping Table**
| Common Schema | File Source | API Source | Notes |
|---------------|-------------|------------|-------|
| `message_id` | `message_id` | `message_id` | Direct mapping |
| `channel_username` | `channel_username` | `channel_username` | Direct mapping |
| `date` | `date` | `date` | ISO timestamp format |
| `text` | `text` | `text` | Direct mapping |
| `creator_username` | `creator_username` | `creator_username` | Direct mapping |
| `creator_first_name` | `creator_first_name` | `creator_first_name` | Direct mapping |
| `creator_last_name` | `creator_last_name` | `creator_last_name` | Direct mapping |
| `media_type` | `media_type` | `media_type` | Direct mapping |
| `file_name` | `file_name` | `file_name` | Direct mapping |
| `file_size` | `file_size` | `file_size` | Integer conversion |
| `mime_type` | `mime_type` | `mime_type` | Direct mapping |
| `caption` | `caption` | `caption` | Direct mapping |
| `views` | `views` | `views` | Integer conversion |
| `forwards` | `forwards` | `forwards` | Integer conversion |
| `replies` | `replies` | `replies` | Integer conversion |
| `is_forwarded` | `is_forwarded` | `is_forwarded` | Boolean conversion |
| `forwarded_from` | `forwarded_from` | `forwarded_from` | Direct mapping |
| `source` | `"file"` | `"api"` | Added during normalization |

#### **Required Fields**
- **Required**: `message_id`, `channel_username`, `date`, `source`
- **Optional but recommended**: `text`, `creator_username`, `media_type`, `file_name`, `file_size`
- **Engagement metrics**: `views`, `forwards`, `replies` (if available)

#### **Validation Rules**
- **message_id**: Must be present and non-empty
- **channel_username**: Must be present and start with '@'
- **date**: Must be valid ISO timestamp format
- **source**: Must be either "file" or "api"
- **file_size**: If present, must be positive integer
- **views/forwards/replies**: If present, must be non-negative integers
- **is_forwarded**: If present, must be boolean

### JSON Output Structure

#### Message Analysis Output
```json
{
  "channel_name": "@SherwinVakiliLibrary",
  "analysis_type": "messages",
  "generated_at": "2024-01-15T10:30:00Z",
  "data_summary": {
    "total_records": 1000,
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "data_quality_score": 0.95
  },
  "analysis_results": {
    "content_statistics": {
      "total_messages": 1000,
      "messages_with_text": 850,
      "empty_messages": 150,
      "text_length": {
        "character_count": {"min": 5, "max": 2000, "mean": 245.6, "median": 180},
        "word_count": {"min": 1, "max": 400, "mean": 48.3, "median": 35},
        "length_categories": {"short": 200, "medium": 500, "long": 150}
      },
      "content_types": {"text_only": 600, "media_only": 100, "text_media": 150, "empty": 150},
      "language_analysis": {
        "primary_language": "English",
        "language_distribution": {"English": 850, "Spanish": 100, "French": 50},
        "total_languages_detected": 3
      }
    },
    "pattern_recognition": {
      "hashtags": {
        "top_hashtags": [{"tag": "#education", "count": 45}, {"tag": "#books", "count": 32}],
        "total_unique_hashtags": 156
      },
      "mentions": {
        "top_mentions": [{"username": "@user123", "count": 23}, {"username": "@admin", "count": 18}],
        "total_unique_mentions": 67
      },
      "urls": {
        "total_urls": 89,
        "top_domains": [{"domain": "example.com", "count": 12}, {"domain": "github.com", "count": 8}]
      },
      "emojis": {
        "top_emojis": [{"emoji": "📚", "count": 34}, {"emoji": "👍", "count": 28}],
        "total_unique_emojis": 45
      }
    },
    "creator_analysis": {
      "top_contributors": [{"username": "user123", "message_count": 156}, {"username": "admin", "message_count": 89}],
      "creator_message_counts": {
        "user123": 156,
        "admin": 89
      }
    }
  },
  "metadata": {
    "source_type": "file",
    "processing_time": "2.5s",
    "version": "1.0.0"
  }
}
```

#### Media Analysis Output
```json
{
  "channel_name": "@SherwinVakiliLibrary",
  "analysis_type": "media",
  "generated_at": "2024-01-15T10:30:00Z",
  "data_summary": {
    "total_records": 1000,
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "data_quality_score": 0.95
  },
  "analysis_results": {
    "file_size_analysis": {
      "storage_usage": {
        "total_storage_bytes": 1073741824,
        "total_storage_mb": 1024,
        "average_file_size_bytes": 1048576,
        "average_file_size_mb": 1.0
      },
      "size_distribution": {
        "categories": {"small": 600, "medium": 300, "large": 100},
        "percentiles": {"50th": 1048576, "75th": 2097152, "90th": 5242880}
      },
      "largest_files": [{"file_name": "large_video.mp4", "size": 52428800}, {"file_name": "presentation.pdf", "size": 20971520}]
    },
    "media_type_analysis": {
      "file_format_distribution": {
        "documents": {"PDF": 200, "DOC": 50, "TXT": 30, "PPT": 20},
        "images": {"JPG": 150, "PNG": 80, "GIF": 20},
        "videos": {"MP4": 40, "AVI": 10},
        "audio": {"MP3": 25, "WAV": 5}
      },
      "mime_type_analysis": {
        "top_mime_types": [{"type": "application/pdf", "count": 200}, {"type": "image/jpeg", "count": 150}]
      },
      "average_sizes_by_type": {"PDF": 1048576, "JPG": 512000, "MP4": 10485760}
    },
    "filename_analysis": {
      "duplicate_filename_detection": {
        "files_with_duplicate_names": 45,
        "total_unique_filenames": 955,
        "total_files": 1000,
        "duplicate_ratio": 0.045,
        "most_common_filenames": [
          {"filename": "document.pdf", "count": 5},
          {"filename": "image.jpg", "count": 4},
          {"filename": "presentation.pptx", "count": 3}
        ]
      },
      "filename_pattern_analysis": {
        "filename_length": {"min": 5, "max": 150, "mean": 28.5, "median": 24},
        "common_extensions": [{"ext": ".pdf", "count": 200}, {"ext": ".jpg", "count": 150}],
        "files_with_special_chars": 23,
        "files_with_spaces": 156
      }
    },
    "filesize_analysis": {
      "duplicate_filesize_detection": {
        "files_with_duplicate_sizes": 78,
        "total_unique_filesizes": 922,
        "total_files": 1000,
        "duplicate_ratio": 0.078,
        "most_common_filesizes": [
          {"size_bytes": 1048576, "count": 8, "size_mb": 1.0},
          {"size_bytes": 2097152, "count": 6, "size_mb": 2.0},
          {"size_bytes": 524288, "count": 5, "size_mb": 0.5}
        ]
      },
      "filesize_distribution_analysis": {
        "size_frequency_distribution": {
          "0-1MB": 600,
          "1-5MB": 300,
          "5-10MB": 80,
          "10MB+": 20
        },
        "potential_duplicates_by_size": [
          {"size_bytes": 1048576, "files": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]},
          {"size_bytes": 2097152, "files": ["img1.jpg", "img2.jpg"]}
        ]
      }
    },
    "media_content_analysis": {
      "caption_analysis": {
        "with_captions": 400,
        "without_captions": 600,
        "caption_length": {"min": 5, "max": 200, "mean": 45, "median": 38}
      },
      "media_engagement": {
        "engagement_by_type": {
          "PDF": {"avg_views": 150, "avg_forwards": 5, "avg_replies": 2},
          "JPG": {"avg_views": 200, "avg_forwards": 8, "avg_replies": 3},
          "MP4": {"avg_views": 300, "avg_forwards": 12, "avg_replies": 5}
        },
        "most_popular_files": [{"file_name": "tutorial.pdf", "views": 500, "forwards": 25, "replies": 10}]
      }
    }
  },
  "metadata": {
    "source_type": "file",
    "processing_time": "3.2s",
    "version": "1.0.0"
  }
}
```

#### Diff Analysis Output
```json
{
  "channel_name": "@SherwinVakiliLibrary",
  "analysis_type": "diff",
  "generated_at": "2024-01-15T10:30:00Z",
  "comparison_summary": {
    "file_records": 1000,
    "api_records": 998,
    "sync_percentage": 99.5
  },
  "differences": {
    "missing_in_api": 5,
    "missing_in_file": 2
  },
  "sync_status": "partially_synced"
}
```

## Configuration

### Environment Configuration
The analysis module will use configuration constants defined in `config.py`:

- **ANALYSIS_BASE**: "reports/analysis" (base directory for analysis reports)
- **COLLECTIONS_DIR**: "reports/collections" (source directory for combined JSON files)
- **COMBINED_COLLECTION_GLOB**: "tg_*_combined.json" (file discovery pattern)
- **API_ENDPOINTS**: Dictionary of API endpoints for data fetching:
  - `channels`: "/api/v1/telegram/channels" (list available channels)
  - `messages`: "/api/v1/telegram/messages" (fetch messages with pagination)
  - `stats`: "/api/v1/telegram/stats" (channel statistics)
- **DEFAULT_DB_URL**: "http://localhost:8000" (API base URL, configurable via environment)
- **DEFAULT_RATE_LIMIT**: 5000 (messages per minute for API calls)
- **DEFAULT_SESSION_COOLDOWN**: 30 (seconds between API sessions)

### Output Configuration
The analysis system generates reports in the following structure following `config.py` patterns:

#### **Base Configuration**
- **Base Directory**: `reports/analysis/` (from `ANALYSIS_BASE`)
- **File Messages Directory**: `reports/analysis/file_messages/` (from `FILE_MESSAGES_DIR`)
- **Files Channels Directory**: `reports/analysis/files_channels/` (from `FILES_CHANNELS_DIR`)

#### **File Naming Patterns**
- **Analysis File Pattern**: `{channel}_analysis.json` (from `ANALYSIS_FILE_PATTERN`)
- **Summary File Pattern**: `analysis_summary.json` (from `ANALYSIS_SUMMARY_PATTERN`)

#### **Directory Structure Rules**
- **Single Channel**: `reports/analysis/file_messages/{channel_name}/`
- **Multiple Channels**: `reports/analysis/file_messages/combined_X_channels/`
- **All Channels**: Individual folders for each discovered channel
- **Channel Name Processing**: Remove '@' prefix from channel names for directory names

#### **Legacy Structure (Deprecated)**
- **Message Reports**: `reports/analysis/messages/` (no longer used)
- **Media Reports**: `reports/analysis/media/` (no longer used)
- **Diff Reports**: `reports/analysis/diff/` (no longer used)

## Error Handling and Validation

### CLI Validation
- **Default Behavior**: All sources (file, API, diff) are enabled by default
- **Individual Source Control**: Each source can be disabled with its respective `--no-*` flag
- **Minimum Source Requirement**: At least one source (file or API) must be enabled
- **Clear Error Messages**: Provide actionable error messages for invalid channel specifications and flag combinations

### Data Validation
- **Schema Validation**: Ensure data structure consistency
- **Quality Checks**: Identify and report data quality issues
- **Source Validation**: Verify data source availability and integrity

### Error Recovery
- **Graceful Degradation**: Continue processing with partial data when possible
  - If API fails, continue with file data only
  - If file parsing fails, skip that file and continue with others
  - If advanced analysis fails, continue with core analysis only
- **Error Reporting**: Comprehensive error logging and reporting
  - Log all errors with timestamps and context
  - Generate error summary in output metadata
  - Provide actionable error messages to users
- **Fallback Strategies**: Alternative approaches when primary methods fail
  - API timeout: Retry with exponential backoff (max 3 retries)
  - File corruption: Skip corrupted files and continue
  - Memory exhaustion: Switch to chunked processing automatically

### User Feedback
- **Progress Indicators**: Show processing progress for long operations
  - Display current phase (Discovery, Loading, Analysis, Output)
  - Show percentage complete for each phase
  - Estimate remaining time for operations >30 seconds
- **Error Messages**: Clear, actionable error messages with specific error codes
- **Validation Reports**: Summary of data quality and validation results
  - Data quality score (0-100%)
  - Missing field counts and percentages
  - Validation warnings and recommendations

### Error Taxonomy
| Error Code | Description | Example | Suggested Action |
|------------|-------------|---------|------------------|
| `CLI_NO_SOURCES` | Both file and API sources disabled | `--no-file --no-api` | Enable at least one source |
| `CLI_DIFF_NO_FILE` | Diff enabled but file source disabled | `--no-file` (with diff enabled) | Enable file source or disable diff |
| `CLI_DIFF_NO_API` | Diff enabled but API source disabled | `--no-api` (with diff enabled) | Enable API source or disable diff |
| `API_TIMEOUT` | API request timeout | API response >30 seconds | Check API status, retry |
| `API_CONNECTION_ERROR` | Cannot connect to API | Connection refused | Check API server status |
| `API_HTTP_ERROR` | API returns error status | HTTP 500, 404, etc. | Check API endpoint |
| `FILE_NOT_FOUND` | Combined file not found | File missing from directory | Check file path |
| `FILE_CORRUPTED` | JSON file is malformed | Invalid JSON syntax | Skip file, continue |
| `FILE_EMPTY` | File contains no messages | Empty messages array | Skip file, continue |
| `VALIDATION_MISSING_FIELD` | Required field missing | message_id is null | Log warning, continue |
| `VALIDATION_INVALID_FORMAT` | Field format invalid | date not ISO format | Log warning, continue |
| `ANALYSIS_FAILED` | Analysis computation error | Division by zero | Log error, continue with other analyses |

## Single Module Architecture

The analysis functionality is consolidated into a single `modules/analysis.py` file for an advanced intermediate analysis system. This approach provides several benefits:

### Advantages of Single Module Design

1. **Easier Maintenance**: All analysis code is in one place, making updates and debugging simpler
2. **Reduced Complexity**: No need to manage multiple files and import relationships
3. **Faster Development**: Developers can work on the entire analysis system without switching between files
4. **Simpler Testing**: Single test file covers all analysis functionality
5. **Easier Deployment**: Fewer files to manage and deploy

### Module Organization

The `analysis.py` module will contain:

- **Data Loaders**: File, API, and dual-source data loading classes
- **Analyzers**: Core analysis type implementations (messages, media, diff)
- **Output Formatters**: JSON formatting and file writing utilities
- **Orchestrator**: Main coordination logic for the analysis pipeline
- **Utilities**: Validation, progress tracking, and helper functions

### Code Structure Within the Module

The `modules/analysis.py` file will be organized into logical sections with clear separation:

```python
# Configuration and Constants
# Data Loading Classes (FileDataLoader, ApiDataLoader, DualSourceDataLoader)
# Analysis Classes (MessageAnalyzer, MediaAnalyzer, DiffAnalyzer)
# Output Classes (JsonFormatter, FileManager, MetadataGenerator)
# Orchestration Classes (AnalysisOrchestrator, ProgressTracker, ErrorHandler)
# Utility Functions (validation, helpers, common operations)
# Main Entry Points (create_analysis_config, run_advanced_intermediate_analysis)
```

**Estimated Module Size**: 600-800 lines of code (advanced intermediate analysis system)
**Maintenance Strategy**: Use clear class boundaries and comprehensive docstrings
**Testing Strategy**: Unit tests for individual classes, integration tests for the full pipeline
**Future Expansion**: Advanced analyses will be added as separate modules in future versions

### Implementation Guidance

#### **Development Phases**
1. **Phase 1**: Data loading and validation (FileDataLoader, ApiDataLoader, validation)
2. **Phase 2**: Core analysis implementations (MessageAnalyzer, MediaAnalyzer)
3. **Phase 3**: Diff analysis and output formatting (DiffAnalyzer, JsonFormatter)
4. **Phase 4**: Error handling and user feedback (ErrorHandler, ProgressTracker)

#### **Implementation Details for Complex Features**

**Pattern Recognition Implementation**:
- **Hashtag Analysis**: Use regex pattern `r'#\w+'` to find hashtags, then `value_counts()` for frequency
- **Mention Analysis**: Use regex pattern `r'@\w+'` to find mentions, then `value_counts()` for frequency
- **URL Analysis**: Use regex pattern `r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'` to find URLs, extract domains with `urlparse()`
- **Emoji Analysis**: Use `emoji.emoji_list()` to find emojis, then `value_counts()` for frequency

**Language Detection Implementation**:
- Use `langdetect.detect()` on message text, handle `LangDetectException` for unknown languages
- Group by detected language and count occurrences
- Calculate primary language as the most frequent language
- Include language distribution in analysis results
- Handle cases where language detection fails (fallback to "Unknown")

**Engagement Analysis Implementation**:
- Group by `media_type` and calculate `mean()` for views, forwards, replies
- Use `nlargest(10)` to find most popular files by engagement metrics

**Filename Analysis Implementation**:
- **Duplicate Filename Detection**: Use `value_counts()` on `file_name` column to find duplicates
- **Filename Pattern Analysis**: Use `str.len()` for length analysis, `str.extract()` with regex for extensions
- **Special Character Detection**: Use regex patterns to identify files with spaces, special chars
- **Extension Analysis**: Use `str.split('.')` and `value_counts()` for extension frequency

**Filesize Analysis Implementation**:
- **Duplicate Filesize Detection**: Use `value_counts()` on `file_size` column to find size duplicates
- **Size Distribution**: Use `pd.cut()` to create size bins and count frequencies
- **Potential Duplicates**: Group by `file_size` and collect filenames for each size
- **Size Clustering**: Use statistical methods to identify common size patterns

**API Pagination Implementation**:
- Check `has_more` field in API response
- Use `page` parameter to fetch next page
- Continue until `has_more` is false

**Diff Analysis Implementation**:
- Compare message_id sets between file and API sources using pandas set operations
- Calculate sync_percentage as: `(common_records / total_unique_records) * 100`
- Determine sync_status based on sync_percentage thresholds:
  - synced: sync_percentage == 100
  - partially_synced: 90 <= sync_percentage < 100
  - not_synced: sync_percentage < 90
- Identify missing records using set difference operations:
  - missing_in_api = file_message_ids - api_message_ids
  - missing_in_file = api_message_ids - file_message_ids

#### **Testing Strategy**
- **Unit Tests**: Test each class independently with mock data
- **Integration Tests**: Test full pipeline with sample files and API responses
- **Error Tests**: Test error handling with invalid data and network failures
- **Performance Tests**: Test with large datasets to verify memory management

#### **Code Quality Standards**
- **Documentation**: Every class and method must have docstrings
- **Type Hints**: Use Python type hints for all function parameters and returns
- **Error Handling**: All external operations (file I/O, API calls) must have try/catch blocks
- **Logging**: Use structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)

## Conclusion

The `analysis` command design provides an advanced intermediate analysis system for analyzing Telegram channel data. By delivering pattern recognition, statistical analysis, language detection, and engagement analysis, it addresses the limitations of the old `report` command while providing comprehensive insights and detailed reporting.

The design emphasizes:
- **Advanced Intermediate Analysis System**: Comprehensive analytics with pattern recognition, language detection, and statistical metrics
- **Structured Data Processing**: Field mapping, validation rules, and comprehensive error handling
- **Detailed Output**: JSON reports with comprehensive analysis results and metadata
- **Clear Development Guidance**: Phased implementation approach with testing strategy

The single-module architecture ensures simplicity and maintainability while providing clear internal organization. The system delivers advanced intermediate analysis capabilities including hashtag analysis, mention tracking, URL analysis, emoji detection, creator analysis, engagement analysis, and comprehensive diff analysis.

