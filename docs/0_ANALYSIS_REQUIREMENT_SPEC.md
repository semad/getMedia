# Analysis Command Design Specification

## Overview

The `analysis` command processes Telegram channel data from files and APIs, generating comprehensive analytics reports. All functionality is consolidated into a single module for simplicity and maintainability.

## Design Goals

1. **Comprehensive Analysis**: Pattern recognition, statistical analysis, and detailed insights
2. **Flexible Data Sources**: Support for file-based data and API endpoints
3. **Performance Optimization**: Efficient processing using pandas DataFrames
4. **Structured Output**: JSON reports with comprehensive results and metadata
5. **Error Handling**: Robust error handling and validation throughout the pipeline


## Requirements

### Configuration Management

#### REQ-001: Configuration Requirements

- Use `config.py` for centralized configuration management
- All paths and patterns must follow `config.py` definitions
- Use `ANALYSIS_BASE` for output directory
- Use `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN` for file naming
- Support environment variable overrides

### Functional Requirements

#### REQ-002: Data Source Requirements

- **File Source**: Read from JSON files matching `COMBINED_COLLECTION_GLOB` pattern
- **API Source**: Fetch data from `API_ENDPOINTS` with pagination and rate limiting
- **Error Handling**: Graceful handling of timeouts, connection errors, and data validation

#### REQ-003: Output Requirements

- **Directory Structure**: Create `ANALYSIS_BASE/{channel_name}/` for each channel (remove '@' prefix)
- **File Naming**: Use `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN` for output files
- **Output Types**:
  - Comprehensive analysis: Complete results using `ANALYSIS_FILE_PATTERN`
  - Summary report: High-level metrics using `ANALYSIS_SUMMARY_PATTERN`
- **Metadata**: Include processing timestamps and data quality metrics

### Non-Functional Requirements

#### REQ-004: Performance Requirements
- Use memory-efficient pandas operations with chunked processing for large datasets
- Provide progress indicators for operations longer than 30 seconds
- Support parallel processing for independent analysis tasks
- Handle datasets with up to 10 million records

#### REQ-005: Error Handling Requirements
- Continue processing with partial data when possible
- Provide clear error messages with specific error codes
- Log all errors with timestamps and context
- Implement graceful degradation for non-critical failures

### Configuration Constants

The analysis module uses these constants from `config.py`:

- `ANALYSIS_BASE`: Base directory for analysis reports
- `ANALYSIS_FILE_PATTERN`: Pattern for analysis report files
- `ANALYSIS_SUMMARY_PATTERN`: Pattern for analysis summary files
- `COMBINED_COLLECTION_GLOB`: File discovery pattern for combined collections
- `API_ENDPOINTS`: Dictionary of API endpoints for data fetching
- `DEFAULT_RATE_LIMIT`: Messages per minute for API calls
- `DEFAULT_SESSION_COOLDOWN`: Seconds between API sessions

## Single Module Architecture

All analysis functionality is consolidated into `modules/analysis_processor.py` for simplicity and maintainability.


## Analysis Requirements

### Dashboard Data Requirements

Based on the generated HTML dashboard pages, the following data elements are required for each channel:

#### REQ-006: Channel Summary Metrics
**Required Data:**
- `total_messages`: Total number of messages in the channel
- `total_files`: Total number of files shared in the channel  
- `total_data_size_bytes`: Total size of all files in bytes
- `total_data_size_formatted`: Human-readable size format (e.g., "89.3 GB")
- `forwarded_messages`: Number of forwarded messages

**Data Sources:**
- Message count from collection files
- File count and size aggregation from `file_size` fields
- Forwarded message detection from `is_forwarded` field

#### REQ-007: File Analysis Section

**File Uniqueness Chart (Pie Chart):**
- `unique_files`: Count of files with unique names
- `duplicate_files`: Count of files with duplicate names
- `duplicate_ratio`: Percentage of duplicate files

**File Size Distribution Chart (Bar Chart):**
- Size bins: `["0-1MB", "1-10MB", "10-100MB", "100MB-1GB", "1GB+"]`
- File counts for each size bin
- Background colors for visual distinction

**File Types Distribution Chart (Bar Chart):**
- File extensions: `["pdf", "epub", "zip", "mp4", "tar", "rar", "png", ...]`
- File counts per extension
- Top 10 most common extensions

**File Analysis Details Metrics:**
- `duplicate_ratio`: Percentage of duplicate files
- `files_with_special_chars`: Count of files with special characters in names
- `files_with_spaces`: Count of files with spaces in names

**File Size Analysis Details:**
- `size_duplicate_ratio`: Percentage of files with duplicate sizes
- `files_with_duplicate_sizes`: Count of files sharing the same size

**File Language Detection:**
- `detected_languages`: List of detected languages in file names and content
- `primary_language`: Most common language detected
- `language_confidence`: Confidence score for language detection
- `language_distribution`: Count of files per detected language

#### REQ-008: Message Activity Section

**Message Activity Timeline Chart (Line Chart):**
- Daily message counts over time
- Date labels in "YYYY-MM-DD" format
- Message count data points for each day
- Line styling with fill and tension

**Monthly Message Distribution Chart (Bar Chart):**
- Monthly message counts: `["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]`
- Message counts per month
- Bar styling with consistent colors

**Message Analysis Details:**
- `unique_hashtags`: Count of unique hashtags used
- `unique_mentions`: Count of unique user mentions
- `unique_urls`: Count of unique URLs shared
- `unique_emojis`: Count of unique emojis used
- `text_length_stats`: Statistical analysis of message text length
  - `average_length`: Mean character count
  - `median_length`: Median character count
  - `shortest_message`: Minimum character count
  - `longest_message`: Maximum character count

#### REQ-009: Content Analysis Section

**Top Hashtags Chart (Bar Chart):**
- Hashtag labels: `["#1", "#reading", "#esl", "#english", ...]`
- Usage counts for each hashtag
- Top 10 most used hashtags

**Top Mentions Chart (Bar Chart):**
- Mention labels: `["@eng_maura", "@english_books_maura", ...]`
- Mention counts for each user
- Top 10 most mentioned users

#### REQ-010: Contributors Section

**Top Contributors Chart (Bar Chart):**
- Contributor usernames: `["books_magazine", "user1", "user2", ...]`
- Message counts per contributor
- Bar styling with distinct colors

**Contributor Analysis Details:**
- `total_contributors`: Total number of unique contributors
- `avg_messages_per_contributor`: Average messages per contributor
- `top_contributors`: List of contributors with message counts and percentages
  - `username`: Contributor username
  - `message_count`: Number of messages posted
  - `percentage`: Percentage of total messages

#### REQ-014: File Language Detection Section

**File Language Distribution Chart (Bar Chart):**
- Language labels: `["English", "Spanish", "French", "German", "Chinese", ...]`
- File counts per detected language
- Top 10 most common languages
- Bar styling with distinct colors

**File Language Analysis Details:**
- `total_languages_detected`: Number of unique languages found
- `primary_language`: Most common language with confidence score
- `language_coverage`: Percentage of files with detected language
- `multilingual_files`: Count of files with mixed languages
- `language_confidence_stats`: Statistical analysis of detection confidence
  - `average_confidence`: Mean confidence score across all detections
  - `min_confidence`: Minimum confidence score
  - `max_confidence`: Maximum confidence score
  - `low_confidence_files`: Count of files with confidence < 0.5

**Language Detection Requirements:**
- Analyze both `file_name` and `caption` fields for language detection
- Use language detection libraries (e.g., `langdetect`, `polyglot`)
- Handle mixed-language content appropriately
- Provide confidence scores for each detection
- Support detection of at least 20 major languages
- Handle cases where language cannot be determined

### Analysis Types

#### Filename Analysis
- Duplicate filename detection and counting
- Filename pattern analysis (length, extensions, special characters)
- Quality assessment and standardization recommendations

#### Filesize Analysis
- Duplicate filesize detection
- Size distribution analysis with meaningful bins (0-1MB, 1-10MB, 10-100MB, 100MB-1GB, 1GB+)
- Storage optimization recommendations

#### Message Content Analysis
- Language detection and distribution
- Content pattern analysis (length, structure, quality)
- Engagement analysis (views, reactions, forwards)
- Temporal analysis (posting patterns, peak activity)

#### Advanced Analytics
- Content classification by type (text, media, links)
- Content categorization and themes
- Time-based trend analysis

### Data Processing Requirements

#### REQ-011: Collection Data Processing
**Required Fields from Collection Files:**
- `message_id`: Unique message identifier
- `channel_username`: Channel identifier (remove '@' prefix for analysis)
- `date`: Message timestamp for temporal analysis
- `text`: Message content for text analysis
- `creator_username`: Message author for contributor analysis
- `creator_first_name`: Author's first name
- `creator_last_name`: Author's last name
- `media_type`: Type of media (if any)
- `file_name`: Filename for file analysis
- `file_size`: File size in bytes for size analysis
- `mime_type`: MIME type for file type classification
- `caption`: Media caption text
- `views`: View count for engagement analysis
- `forwards`: Forward count for engagement analysis
- `replies`: Reply count for engagement analysis
- `is_forwarded`: Boolean flag for forwarded message detection
- `forwarded_from`: Source of forwarded message

#### REQ-012: Data Aggregation Requirements
**File Analysis Data:**
- Group files by `file_name` for duplicate detection
- Group files by `file_size` for size-based duplicate detection
- Extract file extensions from `file_name` for type distribution
- Calculate size bins for distribution charts
- Detect languages in `file_name` and `caption` fields
- Group files by detected language for language distribution
- Calculate language confidence scores and statistics

**Message Analysis Data:**
- Group messages by `date` for temporal analysis
- Extract hashtags from `text` using regex pattern `#\w+`
- Extract mentions from `text` using regex pattern `@\w+`
- Extract URLs from `text` using URL detection patterns
- Extract emojis from `text` using emoji detection libraries
- Calculate text length statistics for message content analysis

**Contributor Analysis Data:**
- Group messages by `creator_username` for contributor analysis
- Calculate message counts and percentages per contributor
- Handle missing or null usernames gracefully

#### REQ-013: Chart Data Format Requirements
**Chart.js Compatible Format:**
- All chart data must be in Chart.js compatible JSON format
- Include `type`, `data`, and `options` properties for each chart
- Use consistent color schemes across charts
- Include responsive design options
- Provide proper labels and legends

**Data Validation:**
- Ensure all numeric values are properly formatted
- Handle null/undefined values gracefully
- Provide fallback values for missing data
- Validate data types before chart generation

#### REQ-015: File Language Detection Implementation
**Language Detection Algorithm:**
1. **Primary Detection**: Use `polyglot` library for high-confidence language detection
2. **Fallback Detection**: Use `pycld2` for additional language support
3. **Text Sources**: Analyze both `file_name` and `caption` fields
4. **Confidence Threshold**: Minimum confidence score of 0.3 for valid detection
5. **Mixed Language Handling**: Detect primary language and flag mixed-language files

**Supported Languages:**
- English, Spanish, French, German, Italian, Portuguese, Russian
- Chinese (Simplified/Traditional), Japanese, Korean, Arabic, Hindi
- Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Czech
- Turkish, Greek, Hebrew, Thai, Vietnamese, Indonesian

**Output Format:**
```json
{
  "file_language_analysis": {
    "detected_languages": ["English", "Spanish", "French"],
    "primary_language": "English",
    "language_confidence": 0.95,
    "language_distribution": [
      {"language": "English", "count": 1200, "percentage": 75.0},
      {"language": "Spanish", "count": 300, "percentage": 18.8},
      {"language": "French", "count": 100, "percentage": 6.2}
    ],
    "total_languages_detected": 3,
    "language_coverage": 95.5,
    "multilingual_files": 25,
    "language_confidence_stats": {
      "average_confidence": 0.87,
      "min_confidence": 0.45,
      "max_confidence": 0.99,
      "low_confidence_files": 12
    }
  }
}
```

**Error Handling:**
- Handle cases where language cannot be detected
- Provide fallback to "Unknown" language category
- Log low-confidence detections for manual review
- Handle encoding issues in file names and captions

## Design Constraints

### Technology Constraints
- **Pandas Library**: All data processing must use Pandas for data loading, manipulation, statistical calculations, and vectorized operations
- **Python 3.8+**: Minimum Python version requirement
- **Memory Efficiency**: Large dataset handling through Pandas chunking

### API Requirements
- Support standard HTTP methods with authentication and rate limiting
- Handle pagination, timeouts, and retries for large datasets
- Implement data validation and graceful error handling

## Dependencies

### Required Packages
- **pandas**: >=1.5.0 (data manipulation and analysis)
- **numpy**: >=1.21.0 (numerical operations)
- **requests**: >=2.28.0 (HTTP client for API calls)
- **langdetect**: >=1.0.9 (language detection for messages)
- **polyglot**: >=16.7.4 (advanced language detection with confidence scores)
- **pycld2**: >=0.41 (Compact Language Detector 2 for file language detection)
- **emoji**: >=2.0.0 (emoji analysis)

### Built-in Modules
- json, pathlib, logging, datetime, collections, itertools

## Caveats

### Critical Issues
- **Missing Testing & Validation**: No testing requirements, validation criteria, or quality assurance specifications
- **Vague Performance Specifications**: No specific performance targets (e.g., processing time limits, memory usage limits)
- **Incomplete Error Handling**: No specific error codes, recovery procedures, or logging format specifications
- **Ambiguous API Requirements**: No authentication mechanisms, request/response schemas, or rate limiting specifics
- **Unclear Data Validation**: No input data format validation, data quality checks, or malformed data handling
- **Missing Security Considerations**: No data encryption, access control, or API security measures
- **Incomplete Output Specifications**: No JSON schema definition, required vs optional fields, or output validation criteria

### Implementation Risks
- **No Prioritization or Phasing**: All analysis types presented as equally important without MVP definition
- **Missing Operational Requirements**: No deployment, monitoring, backup, or maintenance procedures
- **Incomplete Dependency Management**: Version ranges without compatibility testing or conflict resolution
- **No Success Criteria**: No acceptance criteria, performance benchmarks, or quality metrics
- **Missing Integration Specifications**: No integration points, data flow specifications, or interface requirements