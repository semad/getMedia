# Analysis Command Design Specification

## Overview

The `analysis` command is a data analysis tool for Telegram channel data, providing comprehensive data analysis capabilities via multiple sources. It supports file-based data, but by default it uses the REST API for fetching data from Telegram's public channels.

## Architecture Overview

### High-Level Components

```
┌─────────────────┐    ┌─────────────────────────────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │         Single Analysis Module          │    │   Output Layer  │
│                 │    │                                         │    │                 │
│ - Command       │───▶│ ┌─────────────┐ ┌─────────────────────┐ │───▶│ - JSON Format   │
│ - Options       │    │ │Data Loading │ │   Analysis Core     │ │    │ - File Writing  │
│ - Validation    │    │ │- File       │ │ - Message           │ │    │ - Summary Gen   │
└─────────────────┘    │ │- API        │ │                     │ │    └─────────────────┘
                       │ │             │ │                     │ │
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
3. **Data Loading**: Source-specific data loaders within the single module (file or API)
4. **Validation**: Data quality checks and validation
5. **Analysis**: Execution of selected analysis types
6. **Output**: JSON formatting and file generation
7. **Metadata**: Generation of analysis metadata and summaries

## CLI Interface

### Command Structure

The analysis command follows the pattern: `python main.py analysis [options]`
- **Required**: None (uses default behavior)
- **Optional**: Channel selection, verbosity, source selection
- **Mutually Exclusive**: `--api` and `--file` cannot both be used

### Command Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--channels` | `-c` | string | `all` | Comma-separated channel list or `all`, default is all |
| `--verbose` | `-v` | flag | `False` | Enable verbose logging |
| `--api` | | flag | `True` | Use API source only, no file source |
| `--file` | | flag | `False` | Use file source only, no API source |
| `--help` | `-h` | flag | | Show help message |


### Usage Examples

```bash
# Default behavior (API with file fallback)
python main.py analysis

# Analysis with specific channels
python main.py analysis --channels @SherwinVakiliLibrary

# Analysis with verbose logging
python main.py analysis --verbose

# Analysis with specific channels and verbose logging
python main.py analysis --channels @SherwinVakiliLibrary --verbose

# Analysis using file source only 
python main.py analysis --file

# Analysis with specific channels using file source only
python main.py analysis --channels @SherwinVakiliLibrary --file
```

## Data Discovery and Sources

The analysis command first determines what data is available using concrete metrics:
- Analysis is done by default via API, unless the API is not available or the `--file` option is specified, then the analysis is from combined message file only.
- When `--file` is specified, only file source is used (API source is disabled).

- **API Discovery**: When API source exists, 
queries available endpoints to find:
  - Available channels (`API_ENDPOINTS['channels']`)
  - Message counts and data ranges from `API_ENDPOINTS['messages']` endpoint
  - API health check - response time and HTTP status
  - Comprehensive data availability check
- Supports async operations for better performance

- **File Discovery**: When file source is enabled, or no API available, scans the combined collections directory to find:
  - Available combined JSON files matching `COMBINED_COLLECTION_GLOB` pattern
  - Channel names extracted from file metadata
  - channel names are filtered for '@'
  - Message counts from file metadata (any files with messages)
  - File sizes and modification dates (no age restrictions)
  - Using pandas read in combined files as DataFrames


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
8 sources    → 4 unique channels  →  Remove duplicates  →  4 channel folders
```

### **Report Generation Methods**

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

## Data Input Formats

### File Source Data Format
The analysis command reads from combined JSON files in the `COLLECTIONS_DIR` directory:

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

**Channels Endpoint** (`API_ENDPOINTS['channels']`):
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

**Messages Endpoint** (`API_ENDPOINTS['messages']`):
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


## JSON Output Structure

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

## Configuration

### Environment Configuration
The analysis module uses configuration constants defined in `config.py`:

- **ANALYSIS_BASE**: `f"{REPORTS_BASE}/analysis"` (base directory for analysis reports)
- **COLLECTIONS_DIR**: `f"{REPORTS_BASE}/collections"` (source directory for combined JSON files)
- **COMBINED_COLLECTION_GLOB**: `"tg_*_combined.json"` (file discovery pattern)
- **API_ENDPOINTS**: Dictionary of API endpoints for data fetching:
  - `channels`: `f"{API_BASE_PATH}/channels"` (list available channels)
  - `messages`: `f"{API_BASE_PATH}/messages"` (fetch messages with pagination)
  - `stats`: `f"{API_BASE_PATH}/stats"` (channel statistics)
- **DEFAULT_DB_URL**: `"http://localhost:8000"` (API base URL, configurable via environment)
- **DEFAULT_RATE_LIMIT**: `5000` (messages per minute for API calls)
- **DEFAULT_SESSION_COOLDOWN**: `30` (seconds between API sessions)

### Output Configuration
The analysis system generates reports in the following structure following `config.py` patterns:

#### **Directory Structure Rules**
- **Single Channel**: `{ANALYSIS_BASE}/{channel_name}/`
- **Multiple Channels**: `{ANALYSIS_BASE}/combined_X_channels/`
- **All Channels**: Individual folders for each discovered channel
- **Channel Name Processing**: Remove '@' prefix from channel names for directory names

## Error Handling and Validation

### Data Validation
- **Schema Validation**: Ensure data structure consistency
- **Quality Checks**: Identify and report data quality issues
- **Source Validation**: Verify data source availability and integrity

### Error Recovery
- **Graceful Degradation**: Continue processing with partial data when possible
- **Fallback Strategies**: Alternative approaches when primary methods fail
  - API timeout: Retry with exponential backoff (max 3 retries)
  - File corruption: Skip corrupted files and continue
  - Memory exhaustion: Switch to chunked processing automatically

### User Feedback
- **Progress Indicators**: Show processing progress for long operations
- **Error Messages**: Clear, actionable error messages with specific error codes
- **Validation Reports**: Summary of data quality and validation results

### Error Taxonomy
| Error Code | Description | Example | Suggested Action |
|------------|-------------|---------|------------------|
| `CLI_MUTUALLY_EXCLUSIVE` | Both --api and --file specified | `--api --file` | Use only one source option |
| `CLI_NO_SOURCES` | Both file and API sources disabled | `--file --api` | Enable at least one source |
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

The analysis functionality is consolidated into a single `modules/analysis_processor.py` file for an advanced intermediate analysis system. This approach provides several benefits:

### Advantages of Single Module Design

1. **Easier Maintenance**: All analysis code is in one place, making updates and debugging simpler
2. **Reduced Complexity**: No need to manage multiple files and import relationships
3. **Faster Development**: Developers can work on the entire analysis system without switching between files
4. **Simpler Testing**: Single test file covers all analysis functionality
5. **Easier Deployment**: Fewer files to manage and deploy

### Module Organization

The `analysis_processor.py` module will contain:

- **Data Loaders**: File and API data loading classes
- **Analyzers**: Core analysis type implementations (messages, media)
- **Output Formatters**: JSON formatting and file writing utilities
- **Orchestrator**: Main coordination logic for the analysis pipeline
- **Utilities**: Validation, progress tracking, and helper functions

### Code Structure Within the Module

The `modules/analysis_processor.py` file will be organized into logical sections with clear separation:

```python
# Configuration and Constants
# Data Loading Classes (FileDataLoader, ApiDataLoader)
# Analysis Classes (MessageAnalyzer, MediaAnalyzer)
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
3. **Phase 3**: Output formatting and file management (JsonFormatter, FileManager)
4. **Phase 4**: Error handling and user feedback (ErrorHandler, ProgressTracker)

#### **Implementation Details for Complex Features**

**Pattern Recognition Implementation**:
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

#### **Testing Strategy**
- **Unit Tests**: Test each class independently with mock data
- **Integration Tests**: Test full pipeline with sample files and API responses
- **Error Tests**: Test error handling with invalid data and network failures
- **Performance Tests**: Test with large datasets to verify memory management


## Conclusion

The `analysis` command design provides an advanced intermediate analysis system for analyzing Telegram channel data. By delivering pattern recognition, statistical analysis, language detection, and engagement analysis, it addresses the limitations of the old `report` command while providing comprehensive insights and detailed reporting.

The design emphasizes comprehensive analytics with pattern recognition, structured data processing, detailed JSON output, and clear development guidance.

The single-module architecture ensures simplicity and maintainability while providing clear internal organization.

