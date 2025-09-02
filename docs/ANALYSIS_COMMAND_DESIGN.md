# Analysis Command Design Specification

## Overview

The `analysis` command is a redesigned version of the old `report` command, providing comprehensive data analysis capabilities for Telegram channel data. It supports multiple data sources (files, database, and diff comparison) and generates detailed JSON reports for various analysis types.

## Design Goals

1. **Modular Architecture**: Clean separation of concerns with dedicated modules for each responsibility
2. **Flexible Data Sources**: Support for file-based data, database API, and diff analysis
3. **Comprehensive Analysis**: Multiple analysis types covering messages, media, temporal patterns, engagement, and network relationships
4. **Performance Optimization**: Efficient data processing using pandas DataFrames
5. **Output Consistency**: JSON-only output format for easy integration and processing
6. **Extensibility**: Easy to add new analysis types and data sources
7. **Error Handling**: Robust error handling and validation throughout the pipeline

## Architecture Overview

### High-Level Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │  Analysis       │    │   Output Layer  │
│                 │    │  Module         │    │                 │
│ - Command       │───▶│ - Data Loading  │───▶│ - JSON Format   │
│ - Options       │    │ - Analysis      │    │ - File Writing  │
│ - Validation    │    │ - Coordination  │    │ - Summary Gen   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Analysis Core  │
                       │                 │
                       │ - Message       │
                       │ - Media         │
                       │ - Temporal      │
                       │ - Engagement    │
                       │ - Network       │
                       └─────────────────┘
```

### Data Flow Architecture

1. **Input**: CLI arguments and configuration
2. **Data Loading**: Source-specific data loaders (file, database, or dual-source)
3. **Validation**: Data quality checks and validation
4. **Analysis**: Execution of selected analysis types
5. **Output**: JSON formatting and file generation
6. **Metadata**: Generation of analysis metadata and summaries

## CLI Interface

### Command Structure

```bash
python main.py analysis <analysis_type> [options]
```

### Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| `messages` | Text message analysis and statistics | Content analysis |
| `media` | Media file analysis with storage insights | Storage optimization |
| `temporal` | Time-based patterns and trends | Activity analysis |
| `engagement` | User interaction and engagement metrics | Popularity, reach analysis |
| `network` | Channel relationships and network analysis | Cross-channel insights |
| `comprehensive` | All analysis types combined | Complete channel overview |

### Command Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--source` | `-s` | string | `file` | Data source: `file`, `db`, `diff` |
| `--channels` | `-c` | string | `all` | Comma-separated channel list or `all` |
| `--verbose` | `-v` | flag | `False` | Enable verbose logging |
| `--help` | `-h` | flag | | Show help message |

### Usage Examples

```bash
# Basic analysis from files
python main.py analysis messages --source file

# Analysis from database
python main.py analysis media --source db

# Diff analysis comparing file vs database
python main.py analysis comprehensive --source diff

# Diff analysis with specific channels
python main.py analysis comprehensive --source diff --channels channel1,channel2

# Diff analysis with verbose logging
python main.py analysis comprehensive --source diff --verbose
```

## Data Sources

### File Source (`--source file`)
- Reads from combined JSON files in `FILES_CHANNELS_DIR`
- Uses pandas for efficient JSON processing
- Supports both small and large files with appropriate strategies

### Database Source (`--source db`)
- Fetches data from REST API endpoints
- Converts API responses to pandas DataFrames
- Supports async operations for better performance

### Diff Source (`--source diff`)
- Loads data from both file and database sources
- Adds source identifiers to distinguish data origins
- Enables comparison and validation between sources

## Analysis Types

### Message Analysis
- **Content Analysis**: Message length, type distribution, language detection
- **Pattern Recognition**: Common phrases, hashtags, mentions
- **Quality Metrics**: Readability scores, content diversity

### Media Analysis
- **File Size Analysis**: Storage usage, size distribution, optimization opportunities
- **Media Type Analysis**: File formats, compression ratios, quality metrics
- **Storage Impact Analysis**: Growth trends, duplicate detection, efficiency metrics

### Temporal Analysis
- **Activity Patterns**: Daily/weekly/monthly activity cycles
- **Trend Analysis**: Long-term engagement and posting patterns
- **Seasonal Analysis**: Time-based content and activity variations

### Engagement Analysis
- **User Activity**: Active users, participation rates, interaction patterns
- **Content Performance**: Popular content types, engagement metrics
- **Community Dynamics**: User relationships, influence patterns

### Network Analysis
- **Channel Relationships**: Cross-channel interactions and patterns
- **Content Sharing**: Shared content, cross-posting analysis
- **Community Structure**: User networks, influence hierarchies

### Diff Analysis
- **Source Comparison**: File vs database data validation
- **Sync Status**: Data consistency and synchronization state
- **Discrepancy Reporting**: Detailed differences and recommendations

## Data Processing Pipeline

### 1. Data Loading Phase
- **Source Selection**: Based on `--source` option
- **Data Discovery**: Locate and identify data sources
- **Loading Strategy**: Optimize loading based on data size and type
- **Initial Validation**: Basic data structure and format checks

### 2. Data Validation Phase
- **Schema Validation**: Ensure data conforms to expected structure
- **Quality Assessment**: Check for missing, duplicate, or invalid data
- **Source Validation**: For diff analysis, ensure both sources are available
- **Error Reporting**: Collect and categorize validation issues

### 3. Analysis Execution Phase
- **Processor Selection**: Choose appropriate analyzers based on type
- **Data Preparation**: Transform data for analysis requirements
- **Analysis Execution**: Run selected analysis algorithms
- **Result Collection**: Gather and structure analysis outputs

### 4. Output Generation Phase
- **JSON Formatting**: Convert results to structured JSON
- **File Organization**: Create appropriate directory structure
- **Summary Generation**: Create consolidated summary reports
- **Metadata Creation**: Generate analysis metadata and timestamps

## Output Structure

### Directory Organization
```
reports/analysis/
├── {analysis_type}/
│   ├── channels/
│   │   ├── {channel_name}/
│   │   │   ├── analysis.json
│   │   │   └── metadata.json
│   │   └── ...
│   ├── summary.json
│   └── metadata.json
└── diff/
    ├── channels/
    │   ├── {channel_name}/
    │   │   ├── comparison.json
    │   │   └── sync_status.json
    │   └── ...
    ├── summary.json
    └── metadata.json
```

### JSON Output Structure

#### Standard Analysis Output
```json
{
  "channel_name": "channel_id",
  "analysis_type": "messages|media|temporal|engagement|network",
  "generated_at": "ISO timestamp",
  "data_summary": {
    "total_records": 1000,
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "data_quality_score": 0.95
  },
  "analysis_results": {
    // Analysis-specific results
  },
  "metadata": {
    "source_type": "file|db|diff",
    "processing_time": "2.5s",
    "version": "1.0.0"
  }
}
```

#### Diff Analysis Output
```json
{
  "channel_name": "channel_id",
  "analysis_type": "diff",
  "generated_at": "ISO timestamp",
  "comparison_summary": {
    "file_records": 1000,
    "db_records": 998,
    "matching_records": 995,
    "sync_percentage": 99.5
  },
  "differences": {
    "missing_in_db": 5,
    "missing_in_file": 2,
    "data_discrepancies": 3
  },
  "sync_status": "mostly_synced",
  "recommendations": [
    "Sync missing records from file to database",
    "Verify data integrity for discrepant records"
  ]
}
```

## Configuration

### Environment Configuration
The analysis module will use configuration constants defined in `config.py`:

- **DEFAULT_BATCH_SIZE**: Processing batch size for large datasets
- **MAX_FILE_SIZE_MB**: Threshold for large file handling strategies
- **CHUNK_SIZE**: Manual chunking size for large files
- **OUTPUT_DIR**: Base directory for analysis reports
- **ENABLE_LOGGING**: Logging configuration
- **LOG_LEVEL**: Verbosity level for analysis operations

### Analysis Configuration
- **Batch Processing**: Configurable batch sizes for large datasets
- **File Handling**: Strategies for different file sizes and types
- **Output Options**: Directory structure and file naming conventions
- **Logging**: Verbosity levels and log output configuration

## Error Handling and Validation

### Data Validation
- **Schema Validation**: Ensure data structure consistency
- **Quality Checks**: Identify and report data quality issues
- **Source Validation**: Verify data source availability and integrity

### Error Recovery
- **Graceful Degradation**: Continue processing with partial data
- **Error Reporting**: Comprehensive error logging and reporting
- **Fallback Strategies**: Alternative approaches when primary methods fail

### User Feedback
- **Progress Indicators**: Show processing progress for long operations
- **Error Messages**: Clear, actionable error messages
- **Validation Reports**: Summary of data quality and validation results

## Single Module Architecture

The analysis functionality is consolidated into a single `modules/analysis.py` file for simplicity and maintainability. This approach provides several benefits:

### Advantages of Single Module Design

1. **Easier Maintenance**: All analysis code is in one place, making updates and debugging simpler
2. **Reduced Complexity**: No need to manage multiple files and import relationships
3. **Faster Development**: Developers can work on the entire analysis system without switching between files
4. **Simpler Testing**: Single test file covers all analysis functionality
5. **Easier Deployment**: Fewer files to manage and deploy

### Module Organization

The `analysis.py` module will contain:

- **Data Loaders**: File, database, and dual-source data loading classes
- **Analyzers**: All analysis type implementations (messages, media, temporal, etc.)
- **Output Formatters**: JSON formatting and file writing utilities
- **Orchestrator**: Main coordination logic for the analysis pipeline
- **Utilities**: Validation, progress tracking, and helper functions

### Code Structure Within the Module

The module will be organized into logical sections:

- **Data Loading Classes**: Base loader and source-specific implementations
- **Analysis Classes**: Base analyzer and type-specific implementations
- **Output Classes**: JSON formatting and file management
- **Coordination Classes**: Main orchestrator and pipeline management
- **Utility Functions**: Helper functions and common operations

## Conclusion

The `analysis` command design provides a comprehensive, modular, and extensible solution for analyzing Telegram channel data. By focusing on JSON output, leveraging pandas for performance, and supporting multiple data sources including diff analysis, it addresses the limitations of the old `report` command while maintaining consistency with the existing system architecture.

The design emphasizes clean separation of concerns, robust error handling, and performance optimization, making it suitable for both development and production use. The single-module architecture ensures simplicity and maintainability while the modular design ensures that new analysis types and data sources can be easily added in the future.

