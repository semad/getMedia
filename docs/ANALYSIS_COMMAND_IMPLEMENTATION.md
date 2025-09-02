# Analysis Command Implementation Design

## Overview

This document provides detailed implementation specifications for the `analysis` command, a comprehensive data analysis system for Telegram channel data. The command supports multiple data sources (files, API endpoints, diff comparison) and generates detailed JSON reports for various analysis types.

## Implementation Architecture

### Core Module Structure

```
modules/analysis.py
├── Data Loading Layer
│   ├── BaseDataLoader (ABC)
│   ├── FileDataLoader
│   ├── ApiDataLoader
│   └── DualSourceDataLoader
├── Analysis Layer
│   ├── BaseAnalyzer (ABC)
│   ├── MessageAnalyzer
│   ├── MediaAnalyzer
│   ├── TemporalAnalyzer
│   ├── EngagementAnalyzer
│   ├── NetworkAnalyzer
│   └── DiffAnalyzer
├── Output Layer
│   ├── JsonFormatter
│   ├── FileManager
│   └── MetadataGenerator
└── Orchestration Layer
    ├── AnalysisOrchestrator
    ├── ProgressTracker
    └── ErrorHandler
```

### Data Flow Implementation

```
CLI Input → Validation → Source Selection → Data Loading → 
Data Validation → Analysis Execution → Result Processing → 
Output Generation → File Writing → Metadata Creation
```

## Data Types and Imports

### Required Imports
```python
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import aiohttp
import asyncio

# Configuration imports
from config import (
    ANALYSIS_BASE,
    DEFAULT_DB_URL,
    API_BASE_PATH,
    API_ENDPOINTS,
    ANALYSIS_FILE_PATTERN,
    ANALYSIS_SUMMARY_PATTERN
)
```

### Data Type Definitions
```python
# Type aliases for clarity
ProcessedData = pd.DataFrame
RawData = pd.DataFrame
ValidationResult = Dict[str, Any]
AnalysisResult = Dict[str, Any]
```

## Detailed Component Specifications

### 1. Data Loading Layer

#### BaseDataLoader (Abstract Base Class)
```python
@abstractmethod
class BaseDataLoader:
    def __init__(self, config: AnalysisConfig)
    async def load_data(self) -> pd.DataFrame
    async def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]
    def get_source_info(self) -> Dict[str, Any]
```

#### FileDataLoader Implementation
- **Purpose**: Load data from combined JSON files
- **Strategy**: 
  - Small files (< 100MB): Use `pd.read_json()`
  - Large files (≥ 100MB): Manual chunking with `json.loads()`
- **Output**: Single pandas DataFrame with source metadata
- **Error Handling**: Skip corrupted files, log warnings

#### ApiDataLoader Implementation
- **Purpose**: Fetch data from REST API endpoints
- **Strategy**: Async HTTP requests with aiohttp to existing FastAPI endpoints
- **Available Endpoints** (from config.py API_ENDPOINTS):
  - `GET {API_BASE_PATH}/channels` - Get list of all channels
  - `GET {API_BASE_PATH}/messages` - Get messages with filtering and pagination
  - `GET {API_BASE_PATH}/messages/{message_id}` - Get specific message
  - `GET {API_BASE_PATH}/stats` - Get basic statistics
  - `GET {API_BASE_PATH}/stats/enhanced` - Get enhanced statistics
- **Query Parameters**:
  - `channel_username`: Filter by specific channel
  - `media_type`: Filter by media type
  - `page`: Pagination support (default: 1)
  - `items_per_page`: Items per page (default: 100, max: 1000)
- **Data Structure**:
  - **Channels**: Array of objects with `username` field
  - **Messages**: Paginated response with `data`, `total_count`, `has_more`, `page`, `items_per_page`
  - **Message Fields**: `id`, `message_id`, `channel_username`, `date`, `text`, `media_type`, `file_name`, `file_size`, `mime_type`, `views`, `forwards`, `replies`, `creator_username`, etc.
  - **Stats**: Basic counts and enhanced breakdowns including storage, media types, and file sizes
- **Output**: DataFrame with API response data
- **Error Handling**: Retry logic, connection timeouts, pagination handling

#### DualSourceDataLoader Implementation
- **Purpose**: Load data from both file and API sources
- **Strategy**: Parallel loading, source identification
- **Output**: DataFrame with `_source` column ('file' or 'api')
- **Use Case**: Diff analysis between sources

### 2. Analysis Layer

#### BaseAnalyzer (Abstract Base Class)
```python
@abstractmethod
class BaseAnalyzer:
    def __init__(self, config: AnalysisConfig)
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]
    def validate_input(self, data: pd.DataFrame) -> bool
    def get_analysis_metadata(self) -> Dict[str, Any]
```

#### MessageAnalyzer Implementation
- **Input**: DataFrame with message columns (text, channel_username, date, creator_username, etc.)
- **Analysis**:
  - Content length distribution
  - User activity patterns (creator analysis)
  - Language detection (if available)
  - Hashtag and mention analysis
  - Message type categorization
  - Channel-specific message patterns
- **Output**: JSON with statistical summaries and patterns

#### MediaAnalyzer Implementation
- **Input**: DataFrame with media columns (file_size, media_type, mime_type, file_name, etc.)
- **Analysis**:
  - File size distribution and trends (current range: 0-152 MB)
  - Media type popularity (documents: 233, photos: 127)
  - MIME type analysis (PDF: 199, EPUB: 33, photos: 127)
  - Storage growth patterns over time
  - File naming patterns and organization
  - Storage optimization recommendations
- **Output**: JSON with storage insights and recommendations

#### TemporalAnalyzer Implementation
- **Input**: DataFrame with timestamp columns (date, created_at, updated_at)
- **Analysis**:
  - Daily/weekly/monthly activity cycles
  - Peak activity identification
  - Seasonal patterns
  - Trend analysis over time
  - Channel activity patterns over time
- **Output**: JSON with temporal patterns and trends

#### EngagementAnalyzer Implementation
- **Input**: DataFrame with engagement columns (views, forwards, replies, creator_username, etc.)
- **Analysis**:
  - User participation rates (creator analysis)
  - Content popularity metrics (views, forwards, replies)
  - Community dynamics across channels
  - Influence patterns based on engagement
  - Content performance by media type
  - User contribution analysis
- **Output**: JSON with engagement insights

#### NetworkAnalyzer Implementation
- **Input**: DataFrame with cross-channel data (channel_username, creator_username, media_type, etc.)
- **Analysis**:
  - Channel relationships and interactions
  - Content sharing patterns across channels
  - User networks and creator relationships
  - Community structure and dynamics
  - Cross-channel content analysis
- **Output**: JSON with network insights

#### DiffAnalyzer Implementation
- **Input**: DataFrame with source identifiers
- **Analysis**:
  - Source comparison metrics
  - Data consistency checks
  - Sync status assessment
  - Discrepancy reporting
- **Output**: JSON with comparison results and recommendations

### 3. Output Layer

#### JsonFormatter Implementation
- **Purpose**: Convert analysis results to structured JSON
- **Features**:
  - Consistent output format
  - Nested data structures
  - Metadata inclusion
  - Error handling for malformed data
- **Output**: Well-formatted JSON strings

#### FileManager Implementation
- **Purpose**: Organize and write output files
- **Structure**:
  ```
  {ANALYSIS_BASE}/                    # From config.py
  ├── {analysis_type}/
  │   ├── channels/
  │   │   ├── {channel_name}/
  │   │   │   ├── analysis.json      # Uses ANALYSIS_FILE_PATTERN
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

#### MetadataGenerator Implementation
- **Purpose**: Generate comprehensive metadata for each analysis
- **Content**:
  - Generation timestamp
  - Configuration parameters
  - Processing statistics
  - Data quality metrics
  - Source information

### 4. Orchestration Layer

#### AnalysisOrchestrator Implementation
- **Purpose**: Coordinate the entire analysis pipeline
- **Responsibilities**:
  - Validate CLI inputs
  - Create appropriate data loaders
  - Execute analysis workflows
  - Manage output generation
  - Handle errors gracefully
- **Flow Control**: Sequential processing with progress tracking

#### ProgressTracker Implementation
- **Purpose**: Provide user feedback during long operations
- **Features**:
  - Operation progress bars
  - Time estimates
  - Status messages
  - Error reporting

#### ErrorHandler Implementation
- **Purpose**: Centralized error handling and recovery
- **Strategies**:
  - Graceful degradation
  - Partial result generation
  - Comprehensive error logging
  - User-friendly error messages

## Configuration Management

### AnalysisConfig Class
```python
@dataclass
class AnalysisConfig:
    # Data source configuration
    source_type: str = "file"  # file, api, diff
    source_path: Optional[str] = None
    api_base_url: str = DEFAULT_DB_URL  # From config.py
    api_timeout: int = 30
    
    # Processing configuration
    batch_size: int = 1000
    max_file_size_mb: int = 100
    chunk_size: int = 10000
    
    # Output configuration
    output_dir: str = ANALYSIS_BASE  # From config.py
    include_summary: bool = True
    include_raw_data: bool = False
    
    # Channel configuration
    channel_whitelist: Optional[List[str]] = None  # Defaults to DEFAULT_CHANNEL from config.py
    channel_blacklist: Optional[List[str]] = None
    
    # Logging configuration
    verbose: bool = False
    log_level: str = "INFO"
```

### Environment Configuration
```python
# config.py additions (to be added)
ANALYSIS_CONFIG = {
    "DEFAULT_OUTPUT_DIR": ANALYSIS_BASE,  # Use existing constant
    "DEFAULT_BATCH_SIZE": 1000,
    "MAX_FILE_SIZE_MB": 100,
    "CHUNK_SIZE": 10000,
    "API_BASE_URL": DEFAULT_DB_URL,  # Use existing constant
    "API_TIMEOUT": 30,
    "ENABLE_PROGRESS_BARS": True,
    "LOG_LEVEL": "INFO"
}
```

## Error Handling Strategy

### Data Loading Errors
- **File Not Found**: Skip and continue with available files
- **Corrupted Data**: Log warning, skip problematic records
- **API Failures**: Retry with exponential backoff, fallback to file sources
- **Timeout Issues**: Configurable timeout handling
- **Rate Limiting**: Handle API rate limits gracefully

### Analysis Errors
- **Invalid Data**: Graceful degradation with partial results
- **Memory Issues**: Chunked processing for large datasets
- **Processing Failures**: Continue with other analysis types

### Output Errors
- **File System Issues**: Fallback to alternative locations
- **JSON Serialization**: Handle non-serializable data types
- **Permission Issues**: User-friendly error messages

## Performance Optimizations

### Memory Management
- **Chunked Processing**: Process large datasets in manageable chunks
- **DataFrame Optimization**: Use appropriate dtypes and memory-efficient operations
- **Garbage Collection**: Explicit cleanup of large objects

### I/O Optimization
- **Async Operations**: Non-blocking API and file operations
- **Batch Processing**: Efficient handling of multiple files/sources
- **Streaming Writes**: Write output files incrementally
- **API Pagination**: Handle large datasets through pagination

### Processing Efficiency
- **Pandas Operations**: Leverage vectorized operations where possible
- **Selective Analysis**: Only run requested analysis types
- **Caching**: Cache intermediate results for repeated operations

## Testing Strategy

### Unit Testing
- **Component Testing**: Test each analyzer independently
- **Mock Testing**: Use mocks for external dependencies
- **Edge Cases**: Test boundary conditions and error scenarios

### Integration Testing
- **End-to-End Workflows**: Test complete analysis pipelines
- **Data Source Integration**: Verify all data source connections
- **Output Validation**: Ensure correct file generation

### Performance Testing
- **Load Testing**: Test with large datasets
- **Memory Testing**: Monitor memory usage patterns
- **Speed Testing**: Measure processing times

## Deployment Considerations

### Dependencies
```python
# requirements.txt additions (already present in project)
pandas>=1.5.0      # Already used in existing modules
numpy>=1.21.0      # Already used in existing modules
aiohttp>=3.8.0     # For async HTTP requests to API endpoints
click>=8.0.0       # Already used in main.py CLI
pathlib2>=2.3.0    # Already used in existing modules

# Already available in project
# asyncio          # Built-in Python module
# json             # Built-in Python module
# logging          # Built-in Python module
# pathlib          # Built-in Python module
```

### File Permissions
- Ensure write access to `reports/analysis/` directory
- Handle permission errors gracefully
- Provide clear error messages for permission issues

### Logging Configuration
- Configure appropriate log levels
- Implement log rotation for production use
- Ensure logs are written to accessible locations

## Future Extensibility

### Adding New Analysis Types
1. Create new analyzer class inheriting from `BaseAnalyzer`
2. Implement required `analyze()` method
3. Add to orchestrator's processor factory
4. Update CLI help and validation

### Adding New Data Sources
1. Create new loader class inheriting from `BaseDataLoader`
2. Implement required loading methods
3. Add to orchestrator's loader factory
4. Update configuration validation

### Adding New Output Formats
1. Create new formatter class
2. Implement format-specific output logic
3. Add to output manager
4. Update CLI options

## Success Metrics

### Functional Requirements
- [ ] All analysis types execute successfully
- [ ] Data loading works for all source types
- [ ] Output files are generated correctly
- [ ] Error handling works as expected

### Performance Requirements
- [ ] Handle datasets up to 10GB (current: 2.79 GB)
- [ ] Process time under 5 minutes for 1GB datasets
- [ ] Memory usage under 2GB for large operations
- [ ] Async operations complete without blocking
- [ ] Handle API pagination efficiently (current: 400 messages, 100 per page)

### Quality Requirements
- [ ] Comprehensive error logging
- [ ] User-friendly error messages
- [ ] Consistent output format
- [ ] Proper metadata generation

## Configuration Consistency

### Integration with Existing config.py
- **Output Directories**: Uses `ANALYSIS_BASE` from config.py
- **API Configuration**: Uses `DEFAULT_DB_URL` and `API_ENDPOINTS` from config.py
- **File Patterns**: Uses `ANALYSIS_FILE_PATTERN` and `ANALYSIS_SUMMARY_PATTERN`
- **Channel Configuration**: Integrates with existing `DEFAULT_CHANNEL` and `OTHER_CHANNELS`

### Directory Structure Alignment
- **File Analysis**: Outputs to `{ANALYSIS_BASE}/file_messages/channels/{channel_name}/`
- **API Analysis**: Outputs to `{ANALYSIS_BASE}/api_messages/channels/{channel_name}/`
- **Diff Analysis**: Outputs to `{ANALYSIS_BASE}/diff/channels/{channel_name}/`
- **Consistent with**: Existing `FILE_MESSAGES_DIR` and `DB_MESSAGES_DIR` structure

## Data Source Consistency

### Source Types
- **File Source**: JSON files from `FILES_CHANNELS_DIR` (existing data)
- **API Source**: REST endpoints from FastAPI server (current data)
- **Diff Source**: Combination of both sources for comparison

### Data Structure Consistency
- **Input**: All sources convert to pandas DataFrame
- **Processing**: DataFrame operations throughout the pipeline
- **Output**: Consistent JSON structure regardless of source
- **Metadata**: Source information preserved for traceability

## API Data Insights

### Current Data Characteristics
Based on API testing, the system contains:
- **Total Messages**: 400 messages across 4 channels
- **Storage**: 2.79 GB total storage
- **Media Breakdown**: 233 documents, 127 photos
- **Channel Distribution**: 100 messages per channel
- **File Types**: Primarily PDFs (199), EPUBs (33), and photos
- **Date Range**: From 2021-09-26 to 2025-09-01
- **File Sizes**: Average 7.92 MB, range from 0 to ~152 MB

### Data Quality Observations
- **Rich Media Information**: File names, sizes, MIME types, and engagement metrics
- **Engagement Data**: Views, forwards, and replies available for analysis
- **Creator Information**: Username, first name, last name for user analysis
- **Temporal Data**: Precise timestamps for trend analysis
- **Channel Metadata**: Consistent channel identification

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1-2)
- Implement base classes and interfaces
- Create data loading layer
- Set up basic orchestration

### Phase 2: Analysis Implementations (Week 3-4)
- Implement all analyzer classes
- Add error handling and validation
- Create output formatting

### Phase 3: Integration and Testing (Week 5-6)
- Integrate with CLI system
- Comprehensive testing
- Performance optimization

### Phase 4: Documentation and Deployment (Week 7)
- Final documentation updates
- Deployment preparation
- User acceptance testing
