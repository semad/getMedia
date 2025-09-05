# Analysis Command Implementation Plan (Revised)

## Overview

This document provides a comprehensive implementation plan for building the `analysis` command based on the updated requirements in `0_ANALYSIS_REQUIREMENT_SPEC.md`, general design in `1_ANALYSIS_GENERAL_DESIGN.md`, and detailed design in `2_ANALYSIS_DETAIL_DESIGN.md`. This revised plan reflects the single-module architecture, file-source default behavior, and advanced analysis capabilities.

## Implementation Strategy
- **Timeline**: 18 days for production-quality implementation
- **Architecture**: Single module with file source default
- **Tooling**: Using `uv` for package management
- **Quality**: Each phase must meet strict completion criteria

## Implementation Phases

### Phase 1: Foundation Setup (Days 1-2)
**Goal**: Establish the basic structure, configuration, and development environment
**Dependencies**: None (starting phase)
**Completion Criteria**: All foundation components working and tested

#### 1.1 Development Environment Setup
```bash
# Install dependencies
uv add pandas>=1.5.0 pydantic>=2.0.0 aiohttp>=3.8.0 psutil>=5.9.0
uv add langdetect>=1.0.9 emoji>=2.0.0 requests>=2.28.0
uv add --dev pytest>=7.0.0 pytest-asyncio>=0.21.0 pytest-cov>=4.0.0
uv add --dev black>=23.0.0 flake8>=6.0.0 mypy>=1.0.0

# Create project structure
mkdir -p modules tests/{unit,integration,performance}
touch modules/analysis_processor.py modules/__init__.py
touch tests/__init__.py tests/conftest.py
```

#### 1.2 Core Module Structure
```python
# modules/analysis_processor.py
"""
Analysis Command Implementation - Single module containing all functionality
"""

# Core imports
import pandas as pd, logging, asyncio, aiohttp, requests, re, time, gc, psutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import json

# Analysis-specific imports
import langdetect
from langdetect import LangDetectException
import emoji

from pydantic import BaseModel, Field, field_validator, ValidationError
from config import (
    API_ENDPOINTS, COMBINED_COLLECTION_GLOB, COLLECTIONS_DIR, ANALYSIS_BASE,
    ANALYSIS_FILE_PATTERN, ANALYSIS_SUMMARY_PATTERN, DEFAULT_RATE_LIMIT,
    DEFAULT_SESSION_COOLDOWN, DEFAULT_DB_URL
)

# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MEMORY_LIMIT = 100000
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
```

#### 1.3 Configuration Implementation
```python
class AnalysisConfig(BaseModel):
    """Configuration for analysis execution with comprehensive validation."""
    enable_file_source: bool = True
    enable_api_source: bool = False  # Default to file source
    channels: List[str] = Field(default_factory=list)
    verbose: bool = False
    output_dir: str = ANALYSIS_BASE
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = DEFAULT_TIMEOUT
    items_per_page: int = 100
    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY
    
    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v and not all(ch.startswith('@') for ch in v):
            raise ValueError("Channel names must start with '@'")
        return v
    
    @field_validator('api_base_url')
    @classmethod
    def validate_api_url(cls, v):
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            return v
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid URL format")
    
    @field_validator('api_timeout')
    @classmethod
    def validate_timeout(cls, v):
        if not 1 <= v <= 300:
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v
    
    @field_validator('items_per_page')
    @classmethod
    def validate_items_per_page(cls, v):
        if not 1 <= v <= 1000:
            raise ValueError("Items per page must be between 1 and 1000")
        return v
    
    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if not 1000 <= v <= 50000:
            raise ValueError("Chunk size must be between 1000 and 50000")
        return v
    
    @field_validator('memory_limit')
    @classmethod
    def validate_memory_limit(cls, v):
        if not 10000 <= v <= 1000000:
            raise ValueError("Memory limit must be between 10000 and 1000000 messages")
        return v
    
    def validate_config(self) -> bool:
        """Comprehensive configuration validation."""
        try:
            # Check data source configuration
            if not (self.enable_file_source or self.enable_api_source):
                logger.error("At least one data source must be enabled")
                return False
            
            # Validate output directory
            output_path = Path(self.output_dir)
            if not output_path.parent.exists():
                logger.error(f"Output directory parent does not exist: {output_path.parent}")
                return False
            
            # Validate config.py imports
            try:
                from config import API_ENDPOINTS, COMBINED_COLLECTION_GLOB, COLLECTIONS_DIR, ANALYSIS_BASE
                if not API_ENDPOINTS or not isinstance(API_ENDPOINTS, dict):
                    logger.error("API_ENDPOINTS not properly configured")
                    return False
            except ImportError as e:
                logger.error(f"Failed to import config.py: {e}")
                return False
            
            # Validate performance parameters
            if self.chunk_size > self.memory_limit:
                logger.warning("Chunk size larger than memory limit, adjusting")
                self.chunk_size = min(self.chunk_size, self.memory_limit // 10)
            
            logger.info("Configuration validation successful")
            return True
            
        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Configuration validation error: {e}")
            return False
```

**Phase 1 Deliverables**:
- [ ] Complete development environment setup
- [ ] Project structure with proper organization
- [ ] Comprehensive logging configuration
- [ ] Full `AnalysisConfig` with validation
- [ ] Performance monitoring setup
- [ ] All imports and dependencies working
- [ ] Basic unit tests for configuration

### Phase 2: Data Models (Days 3-4)
**Goal**: Implement all Pydantic data models with comprehensive validation
**Dependencies**: Phase 1 complete (foundation setup)
**Completion Criteria**: All models implemented, tested, and validated

#### 2.1 Core Models Implementation
```python
# Data Source Model
class DataSource(BaseModel):
    """Represents a data source for analysis with comprehensive metadata."""
    source_type: str = Field(..., pattern="^(file|api)$")
    channel_name: str = Field(..., min_length=1, max_length=100)
    total_records: int = Field(ge=0, description="Total number of records")
    date_range: Tuple[Optional[datetime], Optional[datetime]] = Field(
        ..., description="Start and end dates of data"
    )
    quality_score: float = Field(ge=0.0, le=1.0, description="Data quality score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata about the source"
    )
    
    @field_validator('channel_name')
    @classmethod
    def validate_channel_name(cls, v):
        if not v.startswith('@'):
            raise ValueError("Channel name must start with '@'")
        if len(v) < 2:
            raise ValueError("Channel name too short")
        return v
    
    @field_validator('date_range')
    @classmethod
    def validate_date_range(cls, v):
        start, end = v
        if start and end and start > end:
            raise ValueError("Start date must be before end date")
        return v

# Message Record Model
class MessageRecord(BaseModel):
    """Individual message record with comprehensive validation."""
    message_id: int = Field(..., ge=1, description="Unique message identifier")
    channel_username: str = Field(..., min_length=1, description="Channel username")
    date: datetime = Field(..., description="Message timestamp")
    text: Optional[str] = Field(None, max_length=4096, description="Message text")
    media_type: Optional[str] = Field(None, description="Type of media if present")
    file_name: Optional[str] = Field(None, max_length=255, description="Filename if media")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    views: Optional[int] = Field(None, ge=0, description="View count")
    forwards: Optional[int] = Field(None, ge=0, description="Forward count")
    replies: Optional[int] = Field(None, ge=0, description="Reply count")
    is_forwarded: Optional[bool] = Field(None, description="Is message forwarded")
    forwarded_from: Optional[str] = Field(None, description="Original channel if forwarded")
    source: str = Field(..., pattern="^(file|api)$", description="Data source type")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Message date cannot be in the future")
        if v < datetime(2020, 1, 1):  # Reasonable lower bound
            raise ValueError("Message date too old")
        return v
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        if v is not None and v > 100 * 1024 * 1024:  # 100MB limit
            logger.warning(f"Large file detected: {v} bytes")
        return v

# Analysis Result Models
class FilenameAnalysisResult(BaseModel):
    """Results of filename analysis with detailed statistics."""
    duplicate_filename_detection: Dict[str, Any] = Field(
        ..., description="Duplicate filename analysis results"
    )
    filename_pattern_analysis: Dict[str, Any] = Field(
        ..., description="Filename pattern analysis results"
    )
    
    @field_validator('duplicate_filename_detection')
    @classmethod
    def validate_duplicate_detection(cls, v):
        required_keys = ['files_with_duplicate_names', 'total_unique_filenames', 
                        'total_files', 'duplicate_ratio', 'most_common_filenames']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key: {key}")
        return v

class FilesizeAnalysisResult(BaseModel):
    """Results of filesize analysis with distribution data."""
    duplicate_filesize_detection: Dict[str, Any] = Field(
        ..., description="Duplicate filesize analysis results"
    )
    filesize_distribution_analysis: Dict[str, Any] = Field(
        ..., description="Filesize distribution analysis results"
    )

class MessageAnalysisResult(BaseModel):
    """Results of message analysis with comprehensive statistics."""
    content_statistics: Dict[str, Any] = Field(
        ..., description="Content analysis statistics"
    )
    pattern_recognition: Dict[str, Any] = Field(
        ..., description="Pattern recognition results"
    )
    creator_analysis: Dict[str, Any] = Field(
        ..., description="Creator analysis results"
    )
    language_analysis: Dict[str, Any] = Field(
        ..., description="Language analysis results"
    )

class AnalysisResult(BaseModel):
    """Complete analysis results with metadata."""
    source: str = Field(..., pattern="^(file|api)$")
    channel_name: str = Field(..., min_length=1)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    total_records_analyzed: int = Field(ge=0)
    filename_analysis: Optional[FilenameAnalysisResult] = None
    filesize_analysis: Optional[FilesizeAnalysisResult] = None
    message_analysis: Optional[MessageAnalysisResult] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    @field_validator('analysis_timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        if v > datetime.now():
            raise ValueError("Analysis timestamp cannot be in the future")
        return v
```

#### 2.2 Model Testing
```python
# tests/unit/test_data_models.py
import pytest
from datetime import datetime, timedelta
from modules.analysis_processor import DataSource, MessageRecord, AnalysisResult

class TestDataSource:
    def test_valid_data_source(self):
        source = DataSource(
            source_type="file",
            channel_name="@test_channel",
            total_records=100,
            date_range=(datetime.now() - timedelta(days=30), datetime.now()),
            quality_score=0.95
        )
        assert source.source_type == "file"
        assert source.channel_name == "@test_channel"
    
    def test_invalid_channel_name(self):
        with pytest.raises(ValueError, match="Channel name must start with '@'"):
            DataSource(
                source_type="file",
                channel_name="invalid_channel",
                total_records=100,
                date_range=(datetime.now(), datetime.now()),
                quality_score=0.95
            )
```

**Phase 2 Deliverables**:
- [ ] All Pydantic models implemented with comprehensive validation
- [ ] Model validation tests with edge cases
- [ ] Sample data generation utilities
- [ ] Model serialization/deserialization tests
- [ ] Performance tests for model operations
- [ ] Documentation for all model fields and validation rules

### Phase 3: Data Loading Classes (Days 5-7)
**Goal**: Implement file and API data loaders with async operations
**Dependencies**: Phase 2 complete (data models)
**Completion Criteria**: Both loaders working with proper async/sync handling

#### 3.1 Data Loading Implementation
- **FileDataLoader**: File discovery, chunked processing, memory optimization
- **ApiDataLoader**: Async operations, pagination, retry logic
- **BaseDataLoader**: Common functionality, error handling, performance monitoring

#### 3.1 BaseDataLoader Implementation
```python
class BaseDataLoader:
    """Base class for all data loaders with common functionality."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
        
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by converting data types."""
        self.logger.info(f"Optimizing DataFrame memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Convert integer columns to smaller types
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Convert float columns to smaller types
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        self.logger.info(f"Optimized DataFrame memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame structure and content."""
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False
        
        # Check required columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values in critical columns
        critical_columns = ['message_id', 'channel_username', 'date']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                self.logger.warning(f"Null values found in critical column: {col}")
        
        # Check data types
        if 'message_id' in df.columns and not pd.api.types.is_integer_dtype(df['message_id']):
            self.logger.error("message_id must be integer type")
            return False
        
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            self.logger.error("date must be datetime type")
            return False
        
        self.logger.info(f"DataFrame validation successful: {len(df)} records")
        return True
    
    def _handle_loading_error(self, error: Exception, context: str) -> None:
        """Handle loading errors with appropriate logging and recovery."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        if isinstance(error, (pd.errors.EmptyDataError, pd.errors.ParserError)):
            self.logger.error(f"Data parsing error in {context}: {error_msg}")
        elif isinstance(error, (OSError, FileNotFoundError)):
            self.logger.error(f"File system error in {context}: {error_msg}")
        elif isinstance(error, MemoryError):
            self.logger.error(f"Memory error in {context}: {error_msg}")
            # Suggest memory optimization
            self.logger.info("Consider reducing chunk_size or memory_limit in configuration")
        elif isinstance(error, (ValueError, TypeError)):
            self.logger.error(f"Data validation error in {context}: {error_msg}")
        else:
            self.logger.error(f"Unexpected error in {context}: {error_type}: {error_msg}")
        
        # Log performance metrics for debugging
        if hasattr(self, 'performance_monitor'):
            stats = self.performance_monitor.get_stats()
            self.logger.debug(f"Performance stats at error: {stats}")
    
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data from source - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_data method")
    
    def discover_sources(self) -> List[str]:
        """Discover available data sources - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement discover_sources method")
```

#### 3.2 Data Loader Testing
```python
# tests/unit/test_data_loaders.py
import pytest
import pandas as pd
from modules.analysis_processor import BaseDataLoader, AnalysisConfig

class TestBaseDataLoader:
    def test_memory_optimization(self):
        config = AnalysisConfig()
        loader = BaseDataLoader(config)
        df = pd.DataFrame({'id': [1, 2, 3], 'category': ['A', 'B', 'A']})
        optimized_df = loader._optimize_dataframe_memory(df.copy())
        assert len(optimized_df) == len(df)
```

**Phase 3 Deliverables**:
- [ ] Complete BaseDataLoader implementation with error handling
- [ ] Memory optimization functionality tested
- [ ] DataFrame validation with comprehensive checks
- [ ] Error handling for all common scenarios
- [ ] Performance monitoring integration
- [ ] Unit tests with edge cases
- [ ] Documentation for all methods

### Phase 4: Analysis Classes (Days 8-12)
**Goal**: Implement all analysis engines with comprehensive algorithms
**Dependencies**: Phase 3 complete (data loading classes)
**Completion Criteria**: All analyzers working with standardized method signatures

#### 4.1 Analysis Classes Implementation
- **FilenameAnalyzer**: Duplicate detection, pattern analysis, quality assessment
- **FilesizeAnalyzer**: Duplicate detection, distribution analysis, size bins
- **LanguageAnalyzer**: Language detection and distribution
- **PatternRecognitionAnalyzer**: Hashtags, mentions, URLs, emojis
- **MessageAnalyzer**: Content statistics, creator analysis, engagement metrics

All analyzers use standardized method signature:
```python
def analyze(self, df: pd.DataFrame, source: DataSource) -> AnalysisResult:
    """Perform analysis with standardized signature."""
```

**Phase 4 Deliverables**:
- [ ] Complete FilenameAnalyzer implementation
- [ ] Complete FilesizeAnalyzer implementation  
- [ ] Complete LanguageAnalyzer implementation
- [ ] Complete PatternRecognitionAnalyzer implementation
- [ ] Complete MessageAnalyzer implementation
- [ ] Standardized method signatures across all analyzers
- [ ] Comprehensive error handling for all analyzers
- [ ] Unit tests for all analysis algorithms

#### 3.3 FileDataLoader Implementation
```python
class FileDataLoader(BaseDataLoader):
    """File-based data loader with chunked processing for large files."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.collections_dir = Path(COLLECTIONS_DIR)
        self.file_pattern = COMBINED_COLLECTION_GLOB
        self.required_columns = [
            'message_id', 'channel_username', 'date', 'text', 
            'media_type', 'file_name', 'file_size', 'views', 
            'forwards', 'replies', 'is_forwarded', 'forwarded_from'
        ]
    
    def discover_sources(self) -> List[str]:
        """Discover available JSON files matching the pattern."""
        if not self.collections_dir.exists():
            return []
        
        matching_files = list(self.collections_dir.glob(self.file_pattern))
        if self.config.channels:
            matching_files = [f for f in matching_files 
                            if any(channel in f.name for channel in self.config.channels)]
        
        return [str(f) for f in matching_files]
    
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data from JSON file with chunked processing for large files."""
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # Large file threshold
            df = self._load_large_file_chunked(file_path)
        else:
            df = pd.read_json(file_path, lines=True)
        
        if not self._validate_dataframe(df, self.required_columns):
            raise ValueError("DataFrame validation failed")
        
        return self._optimize_dataframe_memory(df)
    
    def _load_large_file_chunked(self, file_path: Path) -> pd.DataFrame:
        """Load large file in chunks to manage memory usage."""
        chunks = []
        for chunk_df in pd.read_json(file_path, lines=True, chunksize=self.config.chunk_size):
            chunk_df = self._process_chunk(chunk_df)
            chunks.append(chunk_df)
            
            if len(chunks) * self.config.chunk_size > self.config.memory_limit:
                combined_df = pd.concat(chunks, ignore_index=True)
                combined_df = self._optimize_dataframe_memory(combined_df)
                chunks = [combined_df]
        
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    
    def _process_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Process individual chunk with data cleaning and validation."""
        chunk_df = chunk_df.dropna(how='all')
        if 'date' in chunk_df.columns:
            chunk_df['date'] = pd.to_datetime(chunk_df['date'], errors='coerce')
        
        chunk_df['source'] = 'file'
        for col in ['media_type', 'file_name', 'forwarded_from']:
            chunk_df[col] = chunk_df[col].fillna('')
        for col in ['file_size', 'views', 'forwards', 'replies']:
            chunk_df[col] = chunk_df[col].fillna(0)
        chunk_df['is_forwarded'] = chunk_df['is_forwarded'].fillna(False)
        
        return chunk_df
```

**Phase 3 Deliverables**:
- [ ] Complete BaseDataLoader implementation with error handling
- [ ] Complete FileDataLoader implementation
- [ ] Complete ApiDataLoader implementation
- [ ] Chunked processing for large files
- [ ] Source discovery with filtering
- [ ] Data validation and cleaning
- [ ] Memory management and optimization
- [ ] Comprehensive error handling
- [ ] Unit tests with large file scenarios
- [ ] Performance benchmarks

### Phase 5: Output Management (Days 13-14)
**Goal**: Implement comprehensive output generation and management
**Dependencies**: Phase 4 complete (analysis classes)
**Completion Criteria**: Output manager working with standardized file naming

#### 5.1 JsonOutputManager Implementation
- **Pandas JSON Operations**: Use pandas for all JSON I/O
- **Standardized Naming**: `{channel_name}_` prefix for all files
- **File Organization**: Proper directory structure and metadata

#### 5.2 ApiDataLoader Implementation
```python
class ApiDataLoader(BaseDataLoader):
    """API-based data loader with async operations and retry logic."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.endpoints = API_ENDPOINTS
        self.base_url = config.api_base_url
        self.timeout = config.api_timeout
        self.items_per_page = config.items_per_page
        self.retry_attempts = config.retry_attempts
        self.retry_delay = config.retry_delay
        self.max_messages = config.memory_limit
    
    def discover_sources(self) -> List[str]:
        """Discover available API endpoints."""
        available_sources = []
        for endpoint_name, endpoint_config in self.endpoints.items():
            if self._is_endpoint_available(endpoint_name, endpoint_config):
                available_sources.append(endpoint_name)
        return available_sources
    
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data from API with synchronous wrapper."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.load_data_async(source))
            return future.result(timeout=self.timeout * 2)
    
    async def load_data_async(self, source: str) -> pd.DataFrame:
        """Load data from API asynchronously with pagination and retry logic."""
        if source not in self.endpoints:
            raise ValueError(f"Unknown API source: {source}")
        
        endpoint_config = self.endpoints[source]
        all_data = []
        page = 1
        total_loaded = 0
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            while total_loaded < self.max_messages:
                page_data = await self._fetch_page_with_retry(session, endpoint_config, page)
                if not page_data:
                    break
                
                all_data.extend(page_data)
                total_loaded += len(page_data)
                page += 1
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = self._process_api_data(df)
            return self._optimize_dataframe_memory(df)
        else:
            return pd.DataFrame()
    
    async def _fetch_page_with_retry(self, session: aiohttp.ClientSession, 
                                   endpoint_config: dict, page: int) -> List[dict]:
        """Fetch a single page with retry logic."""
        url = f"{self.base_url}{endpoint_config['path']}"
        params = {'page': page, 'per_page': self.items_per_page, **endpoint_config.get('params', {})}
        
        for attempt in range(self.retry_attempts):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', [])
                    elif response.status == 404:
                        return []
                    else:
                        response.raise_for_status()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise e
        return []
    
    def _is_endpoint_available(self, endpoint_name: str, endpoint_config: dict) -> bool:
        """Check if API endpoint is available."""
        try:
            url = f"{self.base_url}{endpoint_config.get('health_path', '/health')}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ValueError):
            return False
    
    def _process_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process API data with cleaning and normalization."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['source'] = 'api'
        for col in ['media_type', 'file_name', 'forwarded_from']:
            df[col] = df[col].fillna('')
        for col in ['file_size', 'views', 'forwards', 'replies']:
            df[col] = df[col].fillna(0)
        df['is_forwarded'] = df['is_forwarded'].fillna(False)
        return df
```

### Phase 6: Main Orchestration (Days 15-16)
**Goal**: Implement main analysis orchestration with concurrent processing
**Dependencies**: Phase 5 complete (output management)
**Completion Criteria**: Main orchestration working with concurrent data loading and analysis

#### 6.1 Main Orchestration Features
- **Concurrent Processing**: Use asyncio.gather for parallel operations
- **Error Recovery**: Graceful handling of failures and partial results
- **Progress Tracking**: Real-time progress monitoring and logging
- **Resource Management**: Memory and CPU resource optimization

### Phase 7: Testing and Validation (Days 17-18)
**Goal**: Comprehensive testing and quality assurance
**Dependencies**: Phase 6 complete (main orchestration)
**Completion Criteria**: All tests passing, performance benchmarks met

#### 7.1 Testing Strategy
- **Unit Tests**: Individual component testing with edge cases
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Memory usage and execution time benchmarks
- **Edge Case Tests**: Error scenarios and boundary conditions


## Quality Gates and Development

### Phase Completion Criteria
Each phase must meet these criteria before proceeding:
1. **Code Quality**: All code passes linting and type checking
2. **Test Coverage**: Minimum 80% test coverage for new code
3. **Documentation**: All public methods have docstrings
4. **Error Handling**: All error paths are handled appropriately
5. **Performance**: No obvious performance bottlenecks

### Development Workflow
```bash
# All development commands use uv run
uv run pytest tests/ -v                    # Run tests
uv run pytest tests/ --cov=modules.analysis_processor  # Run with coverage
uv run black modules/                       # Format code
uv run flake8 modules/                      # Lint code
uv run mypy modules/                        # Type check

# Add new dependencies during development
uv add requests>=2.28.0                    # Add runtime dependency
uv add --dev pytest-mock>=3.10.0          # Add dev dependency
```

## Conclusion

This implementation plan provides a comprehensive roadmap for building the analysis command based on the updated requirements and design documents. The 18-day timeline allows for proper development, testing, and quality assurance.

Key features:
- **Single Module Architecture**: All functionality consolidated in `modules/analysis_processor.py`
- **File Source Default**: Default behavior uses file source with API as optional
- **Advanced Analysis**: Language detection, pattern recognition, comprehensive content analysis
- **Standardized Output**: Consistent file naming with `{channel_name}_` prefix
- **Modern Dependencies**: Includes `langdetect`, `emoji`, `requests` libraries
- **Comprehensive Testing**: Full test coverage strategy

The implementation should proceed smoothly and result in a production-ready system that meets all requirements.
