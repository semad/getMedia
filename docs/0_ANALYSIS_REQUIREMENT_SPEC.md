# Analysis Command Design Specification

## Overview

The `analysis` command is a comprehensive data analysis tool for Telegram channel data, providing advanced analytics capabilities through multiple data sources. It supports both file-based data processing and REST API integration for real-time data fetching from Telegram channels.

## Purpose

This specification defines the requirements, architecture, and implementation guidelines for the analysis command, which serves as the core analytics engine for processing Telegram channel data and generating actionable insights.

## Design Goals

1. **Data Analysis Tool**: Comprehensive data analysis with pattern recognition, statistical analysis, and detailed insights
2. **Single Module Architecture**: All analysis functionality consolidated into one module for simplicity and maintainability
3. **Flexible Data Sources**: Support for file-based data, API endpoint.
4. **Provide Analytics**: Pattern recognition, language detection, engagement analysis, and statistical metrics
5. **Performance Optimization**: Efficient data processing using pandas DataFrames with vectorized operations
6. **Detailed Output**: JSON reports with comprehensive analysis results and metadata
7. **Extensibility**: Easy to add new analysis types and advanced features in future versions
8. **Error Handling**: Comprehensive error handling and validation throughout the pipeline


## Requirements

### Configuration Management

#### REQ-001: Configuration Requirements

- **REQ-001.1**: Use `config.py` for centralized configuration management
- **REQ-001.2**: All input, output, regex patterns, and paths must follow patterns defined in `config.py`
- **REQ-001.3**: Use `REPORTS_BASE` for base directory configuration
- **REQ-001.4**: Use `COMBINED_COLLECTIONS_DIR` for base input directory
- **REQ-001.5**: Use `ANALYSIS_BASE` for base output directory
- **REQ-001.6**: Use `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN` for file naming conventions
- **REQ-001.7**: Ensure directory structure matches configuration requirements
- **REQ-001.8**: Support environment variable overrides for configuration values

### Functional Requirements

#### REQ-002: Data Source Requirements

The analysis command supports multiple data sources:

- **REQ-002.1**: **File Source Support**
  - Read from combined JSON files in `COMBINED_COLLECTIONS_DIR` directory
  - Support files matching `COMBINED_COLLECTION_GLOB` pattern
  - Handle both small and large files efficiently
  - Provide data quality assessment for file sources
  - Support incremental processing for large datasets

- **REQ-002.2**: **API Source Support**
  - Fetch data from REST API endpoints using `API_ENDPOINTS`
  - Support pagination for large datasets
  - Handle API timeouts and connection errors gracefully
  - Provide real-time data availability checks
  - Implement rate limiting using `DEFAULT_RATE_LIMIT` and retry mechanisms

- **REQ-002.3**: **Data Source Configuration**
  - Support configurable data source selection
  - Provide data source validation and health checks

#### REQ-003: Output Requirements

- **REQ-003.1**: **Base Directory Configuration**
  - **ANALYSIS_BASE**: Base directory for analysis reports
  - Support configurable output directory structure
  - Ensure directory creation and permission handling

- **REQ-003.3**: **Individual Channel Directory Structure**
  - Create separate folders for each channel when analyzing all channels
  - Individual folders for each discovered channel in appropriate analysis directory
  - Each channel gets its own directory: `ANALYSIS_BASE/{channel_name}/`
  - Channel name processing: Remove '@' prefix from channel names for directory names

- **REQ-003.4**: **Structured JSON Output**
  - Generate structured JSON output with comprehensive analysis results
  - Summary report: all report analysis in `ANALYSIS_BASE/`
  - Include metadata and processing timestamps
  - Provide data quality metrics in output

- **REQ-003.5**: **Channel-Specific File Naming - Output File Types per Channel**
  - Use channel names in output file names following `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN`
  - Comprehensive report: `ANALYSIS_FILE_PATTERN` - Complete analysis results
  - Filename analysis: `{channel_name}_filename_analysis.json` - Filename patterns and duplicates
  - Filesize analysis: `{channel_name}_filesize_analysis.json` - File size distributions and duplicates
  - Message analysis: `{channel_name}_message_analysis.json` - Message content and engagement analysis
  - Summary report: `ANALYSIS_SUMMARY_PATTERN` - High-level metrics and insights

### Non-Functional Requirements

#### REQ-004: Performance Requirements
- **REQ-004.1**: Use memory-efficient pandas operations for large datasets
- **REQ-004.2**: Support chunked processing for files exceeding available memory
- **REQ-004.3**: Provide progress indicators for operations longer than 30 seconds
- **REQ-004.4**: Optimize processing time for datasets with >100,000 records
- **REQ-004.5**: Support parallel processing for independent analysis tasks
- **REQ-004.6**: Implement caching for repeated analysis operations

#### REQ-005: Error Handling Requirements
- **REQ-005.1**: Continue processing with partial data when possible
- **REQ-005.2**: Provide clear, actionable error messages with specific error codes
- **REQ-005.3**: Log all errors with timestamps and context
- **REQ-005.4**: Generate error summaries in output metadata
- **REQ-005.5**: Implement graceful degradation for non-critical failures
- **REQ-005.6**: Provide recovery mechanisms for interrupted processing

#### REQ-006: Scalability Requirements
- **REQ-006.1**: Handle datasets with up to 10 million records
- **REQ-006.2**: Support concurrent analysis of multiple channels
- **REQ-006.3**: Provide horizontal scaling capabilities
- **REQ-006.4**: Optimize memory usage for large file processing

#### **Report Generation Methods**
- **`_generate_individual_channel_reports`**: Main method for creating individual channel reports
- **`_get_output_paths`**: Generates channel-specific output paths following `config.py` patterns
- **Channel-specific directory creation**: Each channel gets its own folder with all analysis types


### Environment Configuration

The analysis module will use configuration constants defined in `config.py`:

- **ANALYSIS_BASE**: Base directory for analysis reports
- **COLLECTIONS_DIR**: Source directory for combined JSON files
- **RAW_COLLECTIONS_DIR**: Source directory for raw collection files
- **COMBINED_COLLECTION_GLOB**: File discovery pattern for combined collections
- **API_ENDPOINTS**: Dictionary of API endpoints for data fetching:
  - `channels`: `API_ENDPOINTS['channels']` (list available channels)
  - `messages`: `API_ENDPOINTS['messages']` (fetch messages with pagination)
  - `stats`: `API_ENDPOINTS['stats']` (channel statistics)
- **DEFAULT_DB_URL**: API base URL, configurable via environment
- **DEFAULT_RATE_LIMIT**: Messages per minute for API calls
- **DEFAULT_SESSION_COOLDOWN**: Seconds between API sessions

### Output Configuration

#### **Base Configuration**
- **Base Directory**: `ANALYSIS_BASE`

#### **File Naming Patterns**

- **Analysis File Pattern**: `ANALYSIS_FILE_PATTERN`
- **Summary File Pattern**: `ANALYSIS_SUMMARY_PATTERN`

#### **Directory Structure Rules**

- **Single Channel**: `FILE_MESSAGES_DIR/{channel_name}/`
- **All Channels**: Individual folders for each discovered channel
- **Channel Name Processing**: Remove '@' prefix from channel names for directory names

#### **Directory Configuration**
- `ANALYSIS_BASE`: Base directory for all analysis outputs

#### **File Pattern Configuration**
- `ANALYSIS_FILE_PATTERN`: Pattern for analysis report files
- `ANALYSIS_SUMMARY_PATTERN`: Pattern for analysis summary files
- `COMBINED_COLLECTION_GLOB`: Glob pattern for combined collection files
- `ANALYSIS_FILE_GLOB`: Glob pattern for analysis file discovery
- `ANALYSIS_SUMMARY_GLOB`: Glob pattern for summary file discovery

#### **API Configuration**
- `API_ENDPOINTS`: Dictionary containing all API endpoint paths
- `DEFAULT_DB_URL`: Base URL for API connections
- `DEFAULT_RATE_LIMIT`: Rate limiting for API calls
- `DEFAULT_SESSION_COOLDOWN`: Cooldown period between API sessions

#### **Data Source Configuration**
- `COLLECTIONS_DIR`: Source directory for combined JSON files
- `DEFAULT_CHANNEL`: Default channels for analysis

## Single Module Architecture

The single-module architecture ensures simplicity and maintainability while providing clear internal organization. The system delivers advanced intermediate analysis capabilities including hashtag analysis, mention tracking, URL analysis, emoji detection, creator analysis, engagement analysis, and comprehensive diff analysis.
The analysis functionality is consolidated into a single `modules/analysis_processor.py` module file. This approach provides several benefits:
1. **Easier Maintenance**: All analysis code is in one place, making updates and debugging simpler
2. **Reduced Complexity**: No need to manage multiple files and import relationships
3. **Faster Development**: Developers can work on the entire analysis system without switching between files
4. **Simpler Testing**: Single test file covers all analysis functionality
5. **Easier Deployment**: Fewer files to manage and deploy


## Analysis Requirements

### REQ-007: Filename Analysis
The analysis command MUST provide comprehensive filename analysis capabilities including:

- **REQ-007.1**: **Duplicate Filename Detection**
  - Identify files with identical names (exact matches)
  - Count the number of files sharing each filename
  - Generate a list of most common filenames (top 10)
  - Calculate the ratio of unique filenames vs total files
  - Provide actionable data for duplicate file cleanup

- **REQ-007.2**: **Filename Pattern Analysis**
  - Analyze filename length distribution (min, max, mean, median)
  - Identify common filename patterns and extensions
  - Detect files with special characters or unusual naming conventions
  - Categorize files by extension type and frequency
  - Generate statistics on naming convention compliance

- **REQ-007.3**: **Filename Quality Assessment**
  - Identify files with problematic naming patterns
  - Flag files with spaces, special characters, or non-standard formats
  - Provide recommendations for filename standardization
  - Generate quality scores for filename consistency

### REQ-008: Filesize Analysis
The analysis command MUST provide comprehensive filesize analysis capabilities including:

- **REQ-008.1**: **Duplicate Filesize Detection**
  - Identify files with identical sizes (exact byte matches)
  - Count the number of files sharing each filesize
  - Generate a list of most common filesizes (top 10)
  - Calculate the ratio of unique filesizes vs total files
  - Provide potential duplicate file identification based on size

- **REQ-008.2**: **Filesize Distribution Analysis**
  - Create meaningful size bins (0-1MB, 1-5MB, 5-10MB, 10MB+)
  - Analyze filesize frequency distribution
  - Identify size clusters and patterns
  - Generate storage optimization recommendations

### REQ-009: Language Analysis
The analysis command MUST provide comprehensive language analysis capabilities including:

- **REQ-009.1**: **Language Detection**
  - Detect primary language of message content
  - Identify language distribution across all messages
  - Handle unknown or mixed language content gracefully
  - Provide fallback mechanisms for detection failures

- **REQ-009.2**: **Language Statistics**
  - Calculate percentage distribution of detected languages
  - Identify most common languages (top 5)
  - Generate language diversity metrics
  - Provide language-specific content analysis

### REQ-010: Message Content Analysis
The analysis command MUST provide comprehensive message content analysis including:

- **REQ-010.1**: **Content Pattern Analysis**
  - Analyze message length distribution
  - Identify common content patterns and structures
  - Detect spam or low-quality content indicators
  - Generate content quality metrics

- **REQ-010.2**: **Engagement Analysis**
  - Analyze message engagement metrics (views, reactions, forwards)
  - Identify high-performing content patterns
  - Generate engagement trend analysis
  - Provide content optimization recommendations

### REQ-011: Advanced Analytics
The analysis command MUST provide advanced analytics capabilities including:

- **REQ-011.1**: **Temporal Analysis**
  - Analyze posting patterns and frequency
  - Identify peak activity periods
  - Generate time-based trend analysis
  - Provide scheduling optimization insights

- **REQ-011.2**: **Content Classification**
  - Classify content by type (text, media, links, etc.)
  - Identify content categories and themes
  - Generate content taxonomy
  - Provide content discovery features



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

## API Requirements

### REQ-012: API Integration Requirements
The analysis command MUST provide comprehensive API integration capabilities:

- **REQ-012.1**: **REST API Support**
  - Support standard HTTP methods (GET, POST, PUT, DELETE)
  - Handle authentication and authorization
  - Implement rate limiting and throttling
  - Provide API response caching

- **REQ-012.2**: **Data Fetching**
  - Support pagination for large datasets
  - Handle API timeouts and retries
  - Implement data validation and sanitization
  - Provide real-time data synchronization

- **REQ-012.3**: **Error Handling**
  - Handle API errors gracefully
  - Provide fallback mechanisms
  - Log API interactions and errors
  - Generate API health reports

## Dependencies

### Core Dependencies
The analysis module requires the following Python packages:

#### **Data Processing**
- **pandas**: >=1.5.0 (data manipulation and analysis)
- **numpy**: >=1.21.0 (numerical operations)
- **scipy**: >=1.9.0 (statistical analysis)

#### **API and HTTP**
- **aiohttp**: >=3.8.0 (async HTTP client for API calls)
- **requests**: >=2.28.0 (synchronous HTTP client)
- **httpx**: >=0.24.0 (modern HTTP client)

#### **Language and Text Processing**
- **langdetect**: >=1.0.9 (language detection)
- **emoji**: >=2.0.0 (emoji analysis)
- **textblob**: >=0.17.0 (text processing)
- **nltk**: >=3.7 (natural language processing)

#### **Built-in Modules**
- **asyncio**: built-in (async programming support)
- **json**: built-in (JSON processing)
- **pathlib**: built-in (file path handling)
- **logging**: built-in (logging functionality)
- **datetime**: built-in (date/time operations)
- **collections**: built-in (data structures)
- **itertools**: built-in (iterators)

NO validation or testing section in this document