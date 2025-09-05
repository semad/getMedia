"""
Analysis Command Implementation
Single module containing all analysis functionality

Module Organization:
1. Imports and Constants (lines 1-50)
2. Data Models (lines 51-200)
3. Base Classes (lines 201-300)
4. Data Loaders (lines 301-600)
5. Analysis Engines (lines 601-1000)
6. Output Management (lines 1001-1200)
7. Main Orchestration (lines 1201-1400)
8. Utility Functions (lines 1401-1500)
9. CLI Integration (lines 1501-1600)
"""

# Imports and Constants
import pandas as pd
import logging
import asyncio
import aiohttp
import requests
import re
import time
import gc
import psutil
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
    API_ENDPOINTS, 
    COMBINED_COLLECTION_GLOB, 
    COMBINED_DIR, 
    ANALYSIS_BASE,
    ANALYSIS_FILE_PATTERN,
    ANALYSIS_SUMMARY_PATTERN,
    DEFAULT_RATE_LIMIT,
    DEFAULT_SESSION_COOLDOWN,
    DEFAULT_DB_URL
)

# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_MEMORY_LIMIT = 100000  # messages
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.memory_start = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.logger.debug(f"Performance monitoring started: {self.memory_start:.2f} MB")
    
    def stop(self):
        """Stop performance monitoring and return stats."""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_end - self.memory_start
        
        stats = {
            'elapsed_time_seconds': round(elapsed_time, 3),
            'memory_used_mb': round(memory_used, 2),
            'peak_memory_mb': round(memory_end, 2)
        }
        
        self.logger.debug(f"Performance stats: {stats}")
        return stats
    
    def get_stats(self):
        """Get current performance stats without stopping."""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        memory_current = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_current - self.memory_start
        
        return {
            'elapsed_time_seconds': round(elapsed_time, 3),
            'memory_used_mb': round(memory_used, 2),
            'current_memory_mb': round(memory_current, 2)
        }

# Setup logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration for analysis module."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('analysis_processor')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Initialize logger
logger = setup_logging()

# Data Models (Pydantic)
class AnalysisConfig(BaseModel):
    """Configuration for analysis execution with comprehensive validation."""
    enable_file_source: bool = True
    enable_api_source: bool = False  # Default to file source
    channels: List[str] = Field(default_factory=list)
    verbose: bool = False
    output_dir: str = ANALYSIS_BASE
    api_base_url: str = DEFAULT_DB_URL
    api_timeout: int = DEFAULT_TIMEOUT
    items_per_page: int = 100
    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY
    enable_language_detection: bool = False  # Default to disabled for performance
    
    def __init__(self, **data):
        super().__init__(**data)
        # Note: logger is not a field, it's a property
    
    @property
    def logger(self):
        """Get logger instance."""
        return logging.getLogger(__name__)
    
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
        if not 1 <= v <= 300:  # 1 second to 5 minutes
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v
    
    @field_validator('items_per_page')
    @classmethod
    def validate_items_per_page(cls, v):
        if not 1 <= v <= 1000:  # Reasonable pagination limits
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
                self.logger.error("At least one data source must be enabled")
                return False
            
            # Validate output directory
            output_path = Path(self.output_dir)
            if not output_path.parent.exists():
                self.logger.error(f"Output directory parent does not exist: {output_path.parent}")
                return False
            
            # Validate config.py imports
            try:
                from config import API_ENDPOINTS, COMBINED_COLLECTION_GLOB, COLLECTIONS_DIR, ANALYSIS_BASE
                if not API_ENDPOINTS or not isinstance(API_ENDPOINTS, dict):
                    self.logger.error("API_ENDPOINTS not properly configured")
                    return False
            except ImportError as e:
                self.logger.error(f"Failed to import config.py: {e}")
                return False
            
            # Validate performance parameters
            if self.chunk_size > self.memory_limit:
                self.logger.warning("Chunk size larger than memory limit, adjusting")
                self.chunk_size = min(self.chunk_size, self.memory_limit // 10)
            
            self.logger.info("Configuration validation successful")
            return True
            
        except (OSError, ValueError, TypeError) as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

# Data Models (Pydantic)
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

class MessageRecord(BaseModel):
    """Individual message record with comprehensive validation."""
    message_id: Union[int, str] = Field(..., description="Unique message identifier")
    channel_username: str = Field(..., min_length=1, description="Channel username")
    date: datetime = Field(..., description="Message timestamp")
    text: Optional[str] = Field(None, max_length=4096, description="Message text")
    creator_username: Optional[str] = Field(None, description="Creator username")
    creator_first_name: Optional[str] = Field(None, description="Creator first name")
    creator_last_name: Optional[str] = Field(None, description="Creator last name")
    media_type: Optional[str] = Field(None, description="Type of media if present")
    file_name: Optional[str] = Field(None, max_length=255, description="Filename if media")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type if media")
    caption: Optional[str] = Field(None, description="Media caption")
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

# Base Classes
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

class FileDataLoader(BaseDataLoader):
    """File-based data loader with chunked processing for large files."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.collections_dir = Path(COMBINED_DIR)
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
            # Convert channel names to file name patterns
            channel_patterns = []
            for channel in self.config.channels:
                # Convert @channel to tg_channel_combined.json pattern
                channel_clean = channel.replace('@', '')
                pattern = f"tg_{channel_clean}_combined.json"
                channel_patterns.append(pattern)
            
            matching_files = [f for f in matching_files 
                            if any(pattern in f.name for pattern in channel_patterns)]
        
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
            # Load JSON and process the nested structure
            raw_df = pd.read_json(file_path, lines=False)
            df = self._process_chunk(raw_df)
        
        if not self._validate_dataframe(df, self.required_columns):
            raise ValueError("DataFrame validation failed")
        
        return self._optimize_dataframe_memory(df)
    
    def _load_large_file_chunked(self, file_path: Path) -> pd.DataFrame:
        """Load large file in chunks to manage memory usage."""
        # For large files, we'll load the entire file but process it in chunks
        # This is a simplified approach for now
        df = pd.read_json(file_path, lines=False)
        return self._process_chunk(df)
    
    def _process_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Process individual chunk with data cleaning and validation."""
        # Extract messages from the nested structure
        if 'messages' in chunk_df.columns:
            # Use explode to expand nested messages
            messages_df = chunk_df.explode('messages')
            messages_df = messages_df.dropna(subset=['messages'])
            
            # Use pd.json_normalize for efficient conversion
            df = pd.json_normalize(messages_df['messages'])
            
            # Add source column
            df['source'] = 'file'
        else:
            # If no messages column, treat the entire DataFrame as messages
            df = chunk_df.copy()
            df['source'] = 'file'
        
        # Clean up the data
        df = df.dropna(how='all')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        for col in ['media_type', 'file_name', 'forwarded_from']:
            if col in df.columns:
                df[col] = df[col].fillna('')
        for col in ['file_size', 'views', 'forwards', 'replies']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        if 'is_forwarded' in df.columns:
            df['is_forwarded'] = df['is_forwarded'].fillna(False)
        
        return df

# Analysis Classes
class FilenameAnalyzer:
    """Analyzes filename patterns and duplicates."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> FilenameAnalysisResult:
        """Perform filename analysis."""
        # Filter to only records with filenames
        media_df = df[df['file_name'].notna() & (df['file_name'] != '')]
        
        if media_df.empty:
            return FilenameAnalysisResult(
                duplicate_filename_detection={
                    "files_with_duplicate_names": 0,
                    "total_unique_filenames": 0,
                    "total_files": 0,
                    "duplicate_ratio": 0.0,
                    "most_common_filenames": []
                },
                filename_pattern_analysis={
                    "filename_length": {"count": 0},
                    "common_extensions": [],
                    "files_with_special_chars": 0,
                    "files_with_spaces": 0
                }
            )
        
        return FilenameAnalysisResult(
            duplicate_filename_detection=self._analyze_duplicate_filenames(media_df),
            filename_pattern_analysis=self._analyze_filename_patterns(media_df)
        )
    
    def _analyze_duplicate_filenames(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate filenames."""
        filename_counts = df['file_name'].value_counts()
        
        # Find duplicates (count > 1)
        duplicates = filename_counts[filename_counts > 1]
        files_with_duplicates = duplicates.sum()
        total_files = len(df)
        total_unique_filenames = len(filename_counts)
        duplicate_ratio = files_with_duplicates / total_files if total_files > 0 else 0.0
        
        # Most common filenames
        most_common = filename_counts.head(10)
        most_common_list = [
            {"filename": filename, "count": count}
            for filename, count in most_common.items()
        ]
        
        return {
            "files_with_duplicate_names": int(files_with_duplicates),
            "total_unique_filenames": total_unique_filenames,
            "total_files": total_files,
            "duplicate_ratio": round(duplicate_ratio, 3),
            "most_common_filenames": most_common_list
        }
    
    def _analyze_filename_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze filename patterns."""
        filenames = df['file_name']
        
        # Filename length analysis
        filename_lengths = filenames.str.len()
        length_stats = {
            "count": len(filename_lengths),
            "min": int(filename_lengths.min()),
            "max": int(filename_lengths.max()),
            "mean": round(filename_lengths.mean(), 2),
            "median": int(filename_lengths.median())
        }
        
        # Extension analysis
        extensions = filenames.str.extract(r'\.([^.]+)$')[0].str.lower()
        extension_counts = extensions.value_counts()
        common_extensions = [
            {"ext": f".{ext}", "count": count}
            for ext, count in extension_counts.head(10).items()
        ]
        
        # Special character analysis
        special_char_pattern = r'[^\w\s.-]'
        files_with_special_chars = filenames.str.contains(special_char_pattern, regex=True).sum()
        
        # Space analysis
        files_with_spaces = filenames.str.contains(' ').sum()
        
        return {
            "filename_length": length_stats,
            "common_extensions": common_extensions,
            "files_with_special_chars": int(files_with_special_chars),
            "files_with_spaces": int(files_with_spaces)
        }

class FilesizeAnalyzer:
    """Analyzes filesize patterns and duplicates."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> FilesizeAnalysisResult:
        """Perform filesize analysis."""
        # Filter to only records with filesizes
        media_df = df[df['file_size'].notna() & (df['file_size'] > 0)]
        
        if media_df.empty:
            return FilesizeAnalysisResult(
                duplicate_filesize_detection={
                    "files_with_duplicate_sizes": 0,
                    "total_unique_filesizes": 0,
                    "total_files": 0,
                    "duplicate_ratio": 0.0,
                    "most_common_filesizes": []
                },
                filesize_distribution_analysis={
                    "size_frequency_distribution": {},
                    "potential_duplicates_by_size": []
                }
            )
        
        return FilesizeAnalysisResult(
            duplicate_filesize_detection=self._analyze_duplicate_filesizes(media_df),
            filesize_distribution_analysis=self._analyze_filesize_distribution(media_df)
        )
    
    def _analyze_duplicate_filesizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate filesizes."""
        filesize_counts = df['file_size'].value_counts()
        
        # Find duplicates (count > 1)
        duplicates = filesize_counts[filesize_counts > 1]
        files_with_duplicates = duplicates.sum()
        total_files = len(df)
        total_unique_filesizes = len(filesize_counts)
        duplicate_ratio = files_with_duplicates / total_files if total_files > 0 else 0.0
        
        # Most common filesizes
        most_common = filesize_counts.head(10)
        most_common_list = [
            {
                "size_bytes": int(size),
                "count": count,
                "size_mb": round(size / (1024 * 1024), 2)
            }
            for size, count in most_common.items()
        ]
        
        return {
            "files_with_duplicate_sizes": int(files_with_duplicates),
            "total_unique_filesizes": total_unique_filesizes,
            "total_files": total_files,
            "duplicate_ratio": round(duplicate_ratio, 3),
            "most_common_filesizes": most_common_list
        }
    
    def _analyze_filesize_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze filesize distribution."""
        file_sizes = df['file_size']
        
        # Size distribution bins
        size_bins = pd.cut(
            file_sizes,
            bins=[0, 1024*1024, 5*1024*1024, 10*1024*1024, float('inf')],
            labels=['0-1MB', '1-5MB', '5-10MB', '10MB+']
        )
        size_distribution = size_bins.value_counts().to_dict()
        
        # Potential duplicates by size with filtering
        size_groups = df.groupby('file_size')
        
        # Filter groups with more than one file and convert to list efficiently
        duplicate_groups = size_groups.filter(lambda x: len(x) > 1)
        
        if not duplicate_groups.empty:
            # Group by size again and aggregate filenames
            potential_duplicates = (
                duplicate_groups.groupby('file_size')['file_name']
                .apply(list)
                .reset_index()
                .rename(columns={'file_name': 'files'})
            )
            
            # Convert to list of dicts and sort by size
            potential_duplicates = [
                {
                    "size_bytes": int(size),
                    "files": files
                }
                for size, files in zip(potential_duplicates['file_size'], potential_duplicates['files'])
            ]
            
            # Sort by size and limit to top 10
            potential_duplicates.sort(key=lambda x: x['size_bytes'], reverse=True)
            potential_duplicates = potential_duplicates[:10]
        else:
            potential_duplicates = []
        
        return {
            "size_frequency_distribution": size_distribution,
            "potential_duplicates_by_size": potential_duplicates
        }

class LanguageAnalyzer:
    """Analyzes language distribution and detection in messages."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> Dict[str, Any]:
        """Perform language analysis on message text with progress reporting."""
        if df.empty or 'text' not in df.columns:
            return {
                "primary_language": "Unknown",
                "language_distribution": {},
                "total_languages_detected": 0,
                "detection_coverage": 0.0
            }
        
        # Filter out empty or null text
        text_series = df['text'].dropna()
        text_series = text_series[text_series.str.strip() != '']
        
        if text_series.empty:
            return {
                "primary_language": "Unknown",
                "language_distribution": {},
                "total_languages_detected": 0,
                "detection_coverage": 0.0
            }
        
        total_messages = len(text_series)
        self.logger.info(f"🔍 Starting language detection for {total_messages} messages...")
        
        # Use chunked processing for better performance and progress reporting
        languages = self._detect_languages_chunked(text_series, total_messages)
        
        # Count language distribution
        lang_counts = pd.Series(languages).value_counts()
        total_detected = len(languages)
        unknown_count = lang_counts.get("Unknown", 0)
        
        # Calculate coverage (percentage of successfully detected languages)
        detection_coverage = (total_detected - unknown_count) / total_detected if total_detected > 0 else 0.0
        
        # Get primary language (most common, excluding Unknown)
        lang_distribution = lang_counts.to_dict()
        primary_language = "Unknown"
        if len(lang_distribution) > 1 or "Unknown" not in lang_distribution:
            primary_language = lang_counts.index[0] if not lang_counts.empty else "Unknown"
        
        self.logger.info(f"✅ Language detection completed: {total_detected} messages processed, {len(lang_distribution)} languages detected")
        
        return {
            "primary_language": primary_language,
            "language_distribution": lang_distribution,
            "total_languages_detected": len(lang_distribution),
            "detection_coverage": round(detection_coverage, 3)
        }
    
    def _detect_languages_chunked(self, text_series: pd.Series, total_messages: int) -> List[str]:
        """Detect languages using chunked processing with progress reporting."""
        languages = []
        chunk_size = 1000  # Process 1000 messages at a time
        processed = 0
        
        for i in range(0, len(text_series), chunk_size):
            chunk = text_series.iloc[i:i+chunk_size]
            chunk_languages = []
            
            for text in chunk:
                try:
                    # Skip very short texts that cause "No features in text" errors
                    text_str = str(text).strip()
                    if len(text_str) < 3:
                        chunk_languages.append("Unknown")
                        continue
                    
                    lang = langdetect.detect(text_str)
                    chunk_languages.append(lang)
                except (LangDetectException, Exception) as e:
                    # Only log debug for specific errors, not "No features in text"
                    if "No features in text" not in str(e):
                        self.logger.debug(f"Language detection failed for text: {e}")
                    chunk_languages.append("Unknown")
            
            languages.extend(chunk_languages)
            processed += len(chunk)
            
            # Progress reporting every 10% or every 1000 messages
            if processed % max(1000, total_messages // 10) == 0 or processed == total_messages:
                progress_pct = (processed / total_messages) * 100
                self.logger.info(f"📊 Language detection progress: {processed}/{total_messages} messages ({progress_pct:.1f}%)")
        
        return languages

class PatternRecognitionAnalyzer:
    """Analyzes patterns in message content including hashtags, mentions, URLs, and emojis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compiled regex patterns for performance
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> Dict[str, Any]:
        """Perform pattern recognition analysis."""
        if df.empty or 'text' not in df.columns:
            return {
                "hashtags": {"top_hashtags": [], "total_unique_hashtags": 0},
                "mentions": {"top_mentions": [], "total_unique_mentions": 0},
                "urls": {"total_urls": 0, "top_domains": []},
                "emojis": {"top_emojis": [], "total_unique_emojis": 0}
            }
        
        # Filter text content
        text_df = df[df['text'].notna() & (df['text'] != '')]
        
        if text_df.empty:
            return {
                "hashtags": {"top_hashtags": [], "total_unique_hashtags": 0},
                "mentions": {"top_mentions": [], "total_unique_mentions": 0},
                "urls": {"total_urls": 0, "top_domains": []},
                "emojis": {"top_emojis": [], "total_unique_emojis": 0}
            }
        
        return {
            "hashtags": self._analyze_hashtags(text_df),
            "mentions": self._analyze_mentions(text_df),
            "urls": self._analyze_urls(text_df),
            "emojis": self._analyze_emojis(text_df)
        }
    
    def _analyze_hashtags(self, text_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hashtag patterns."""
        all_hashtags = []
        for text in text_df['text']:
            hashtags = self.hashtag_pattern.findall(str(text))
            all_hashtags.extend(hashtags)
        
        if not all_hashtags:
            return {"top_hashtags": [], "total_unique_hashtags": 0}
        
        hashtag_counts = pd.Series(all_hashtags).value_counts()
        top_hashtags = [{"tag": tag, "count": count} for tag, count in hashtag_counts.head(10).items()]
        
        return {
            "top_hashtags": top_hashtags,
            "total_unique_hashtags": len(hashtag_counts)
        }
    
    def _analyze_mentions(self, text_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze mention patterns."""
        all_mentions = []
        for text in text_df['text']:
            mentions = self.mention_pattern.findall(str(text))
            all_mentions.extend(mentions)
        
        if not all_mentions:
            return {"top_mentions": [], "total_unique_mentions": 0}
        
        mention_counts = pd.Series(all_mentions).value_counts()
        top_mentions = [{"username": mention, "count": count} for mention, count in mention_counts.head(10).items()]
        
        return {
            "top_mentions": top_mentions,
            "total_unique_mentions": len(mention_counts)
        }
    
    def _analyze_urls(self, text_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze URL patterns."""
        all_urls = []
        for text in text_df['text']:
            urls = self.url_pattern.findall(str(text))
            all_urls.extend(urls)
        
        if not all_urls:
            return {"total_urls": 0, "top_domains": []}
        
        # Extract domains
        domains = []
        for url in all_urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                if domain:
                    domains.append(domain)
            except Exception:
                continue
        
        if not domains:
            return {"total_urls": len(all_urls), "top_domains": []}
        
        domain_counts = pd.Series(domains).value_counts()
        top_domains = [{"domain": domain, "count": count} for domain, count in domain_counts.head(5).items()]
        
        return {
            "total_urls": len(all_urls),
            "top_domains": top_domains
        }
    
    def _analyze_emojis(self, text_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze emoji patterns."""
        all_emojis = []
        for text in text_df['text']:
            try:
                emojis = emoji.emoji_list(str(text))
                for emoji_data in emojis:
                    all_emojis.append(emoji_data['emoji'])
            except Exception as e:
                self.logger.debug(f"Emoji analysis failed for text: {e}")
                continue
        
        if not all_emojis:
            return {"top_emojis": [], "total_unique_emojis": 0}
        
        emoji_counts = pd.Series(all_emojis).value_counts()
        top_emojis = [{"emoji": emoji_char, "count": count} for emoji_char, count in emoji_counts.head(10).items()]
        
        return {
            "top_emojis": top_emojis,
            "total_unique_emojis": len(emoji_counts)
        }

class MessageAnalyzer:
    """Analyzes message content, patterns, and creators."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.language_analyzer = LanguageAnalyzer(config)
        self.pattern_analyzer = PatternRecognitionAnalyzer(config)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> MessageAnalysisResult:
        """Perform comprehensive message analysis."""
        if df.empty:
            return MessageAnalysisResult(
                content_statistics={},
                pattern_recognition={},
                creator_analysis={},
                language_analysis={}
            )
        
        # Conditionally run language analysis based on config
        if self.config.enable_language_detection:
            language_analysis = self.language_analyzer.analyze(df, source)
        else:
            self.logger.info("⏭️  Skipping language detection (disabled by config)")
            language_analysis = {}
        
        return MessageAnalysisResult(
            content_statistics=self._analyze_content_statistics(df),
            pattern_recognition=self.pattern_analyzer.analyze(df, source),
            creator_analysis=self._analyze_creators(df),
            language_analysis=language_analysis
        )
    
    def _analyze_content_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content statistics."""
        # Text content analysis
        text_df = df[df['text'].notna() & (df['text'] != '')]
        
        if text_df.empty:
            return {
                "total_messages": len(df),
                "messages_with_text": 0,
                "text_length_stats": {"count": 0},
                "media_messages": len(df[df['media_type'].notna()]),
                "forwarded_messages": len(df[df['is_forwarded'] == True]) if 'is_forwarded' in df.columns else 0
            }
        
        # Text length statistics
        text_lengths = text_df['text'].str.len()
        text_length_stats = {
            "count": len(text_lengths),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "mean": round(text_lengths.mean(), 2),
            "median": int(text_lengths.median())
        }
        
        return {
            "total_messages": len(df),
            "messages_with_text": len(text_df),
            "text_length_stats": text_length_stats,
            "media_messages": len(df[df['media_type'].notna()]),
            "forwarded_messages": len(df[df['is_forwarded'] == True]) if 'is_forwarded' in df.columns else 0
        }
    
    def _analyze_creators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze message creators using pandas operations."""
        # Filter messages with creator information
        creator_df = df[df['creator_username'].notna()]
        
        if creator_df.empty:
            return {
                "total_creators": 0,
                "most_active_creators": [],
                "creator_message_stats": {}
            }
        
        # Creator analysis using pandas
        creator_counts = creator_df['creator_username'].value_counts()
        creator_message_stats = {
            "total_creators": len(creator_counts),
            "total_messages": len(creator_df),
            "avg_messages_per_creator": round(len(creator_df) / len(creator_counts), 2)
        }
        
        # Most active creators
        most_active = [
            {
                "username": username,
                "message_count": count,
                "percentage": round((count / len(creator_df)) * 100, 2)
            }
            for username, count in creator_counts.head(10).items()
        ]
        
        return {
            "total_creators": len(creator_counts),
            "most_active_creators": most_active,
            "creator_message_stats": creator_message_stats
        }

# Main entry points
def create_analysis_config(**kwargs) -> AnalysisConfig:
    """Create analysis configuration from kwargs."""
    return AnalysisConfig(**kwargs)

# Output Management
class JsonOutputManager:
    """Manages JSON output operations using pandas."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_analysis_results(self, results: Dict[str, Any], filename: str) -> str:
        """Save analysis results to JSON file using pandas."""
        output_path = self.output_dir / f"{filename}.json"
        
        # Use pandas to_json directly with the results dict
        pd.Series([results]).to_json(
            output_path,
            orient='records',
            indent=2,
            date_format='iso',
            force_ascii=False
        )
        
        self.logger.info(f"Saved analysis results to: {output_path}")
        return str(output_path)
    
    def save_combined_results(self, results_list: List[Dict[str, Any]], filename: str) -> str:
        """Save multiple analysis results to JSON file using pandas."""
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert list of results to DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Use pandas to_json for output
        results_df.to_json(
            output_path,
            orient='records',
            indent=2,
            date_format='iso',
            force_ascii=False
        )
        
        self.logger.info(f"Saved combined results to: {output_path}")
        return str(output_path)
    
    def load_previous_results(self, filename: str) -> pd.DataFrame:
        """Load previous analysis results from JSON file using pandas."""
        file_path = self.output_dir / f"{filename}.json"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        # Use pandas read_json for input
        return pd.read_json(file_path, orient='records')
    
    def compare_results(self, current_results: Dict[str, Any], previous_results: pd.DataFrame) -> Dict[str, Any]:
        """Compare current results with previous results using pandas."""
        if previous_results.empty:
            return {"comparison": "No previous results available"}
        
        # Perform comparison using pandas operations without unnecessary DataFrame creation
        comparison = {
            "current_count": 1,  # Single result being compared
            "previous_count": len(previous_results),
            "difference": 1 - len(previous_results),
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison

# Main Orchestration
async def run_advanced_intermediate_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Main analysis orchestration function."""
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not config.validate_config():
        raise ValueError("Invalid configuration provided")
    
    logger.info("Starting advanced intermediate analysis...")
    
    try:
        # Initialize data loaders
        file_loader = FileDataLoader(config)
        
        # Discover sources
        sources = []
        if config.enable_file_source:
            file_sources = file_loader.discover_sources()
            sources.extend(file_sources)
            logger.info(f"Discovered {len(file_sources)} file sources")
        
        if not sources:
            return {"error": "No data sources available"}
        
        # Filter sources by channels if specified
        if config.channels:
            sources = [s for s in sources if any(channel in s for channel in config.channels)]
            logger.info(f"Filtered to {len(sources)} sources matching specified channels")
        
        results = {}
        
        # Process each source
        total_sources = len(sources)
        for i, source_path in enumerate(sources, 1):
            try:
                logger.info(f"📁 Processing source {i}/{total_sources}: {source_path}")
                
                # Load data
                logger.info(f"📥 Loading data from {source_path}...")
                df = file_loader.load_data(source_path)
                
                if df.empty:
                    logger.warning(f"⚠️ No data found in source: {source_path}")
                    continue
                
                logger.info(f"📊 Loaded {len(df)} messages from {source_path}")
                
                # Create data source object
                source = DataSource(
                    source_type="file",
                    channel_name=Path(source_path).stem.replace('tg_', '@').replace('_combined', ''),
                    total_records=len(df),
                    date_range=(df['date'].min(), df['date'].max()) if 'date' in df.columns else (None, None),
                    quality_score=0.95,  # Default quality score
                    metadata={'file_path': source_path}
                )
                
                # Run analysis
                logger.info(f"🔍 Starting analysis for {source.channel_name}...")
                filename_analyzer = FilenameAnalyzer(config)
                filesize_analyzer = FilesizeAnalyzer(config)
                message_analyzer = MessageAnalyzer(config)
                
                logger.info(f"📝 Running filename analysis...")
                filename_result = filename_analyzer.analyze(df, source)
                
                logger.info(f"📏 Running filesize analysis...")
                filesize_result = filesize_analyzer.analyze(df, source)
                
                logger.info(f"💬 Running message analysis...")
                message_result = message_analyzer.analyze(df, source)
                
                # Generate output files with standardized naming
                channel_name_clean = source.channel_name.replace('@', '')
                output_manager = JsonOutputManager(config)
                
                # Save individual analysis files
                filename_output = output_manager.save_analysis_results(
                    filename_result.dict(), 
                    f"{channel_name_clean}_filename_analysis"
                )
                filesize_output = output_manager.save_analysis_results(
                    filesize_result.dict(), 
                    f"{channel_name_clean}_filesize_analysis"
                )
                message_output = output_manager.save_analysis_results(
                    message_result.dict(), 
                    f"{channel_name_clean}_message_analysis"
                )
                
                # Create comprehensive analysis result
                comprehensive_result = {
                    "channel_name": source.channel_name,
                    "analysis_type": "comprehensive",
                    "generated_at": datetime.now().isoformat(),
                    "data_summary": {
                        "total_records": len(df),
                        "source_type": source.source_type,
                        "processed_at": datetime.now().isoformat()
                    },
                    "analysis_results": {
                        "filename_analysis": filename_result.dict(),
                        "filesize_analysis": filesize_result.dict(),
                        "message_analysis": message_result.dict()
                    }
                }
                
                # Save comprehensive analysis
                comprehensive_output = output_manager.save_analysis_results(
                    comprehensive_result,
                    f"{channel_name_clean}_analysis"
                )
                
                # Store results with file paths
                result = {
                    "filename_analysis": filename_result.dict(),
                    "filesize_analysis": filesize_result.dict(),
                    "message_analysis": message_result.dict(),
                    "output_files": {
                        "filename_analysis": filename_output,
                        "filesize_analysis": filesize_output,
                        "message_analysis": message_output,
                        "comprehensive_analysis": comprehensive_output
                    },
                    "metadata": {
                        "source_type": source.source_type,
                        "total_records": len(df),
                        "processed_at": datetime.now().isoformat()
                    }
                }
                
                results[source.channel_name] = result
                logger.info(f"Completed analysis for {source.channel_name}")
                
            except Exception as e:
                logger.error(f"Error processing source {source_path}: {e}")
                results[source_path] = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "source_path": source_path
                }
        
        logger.info(f"Analysis completed for {len(results)} sources")
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }

# CLI Integration
def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analysis Command")
    parser.add_argument("--channels", "-c", nargs="+", help="Channel names to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--file", action="store_true", help="Use file source only")
    parser.add_argument("--api", action="store_true", help="Use API source only")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_analysis_config(
        channels=args.channels or [],
        verbose=args.verbose,
        enable_file_source=not args.api,
        enable_api_source=not args.file
    )
    
    # Run analysis
    result = asyncio.run(run_advanced_intermediate_analysis(config))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
