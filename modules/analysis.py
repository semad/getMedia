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
import sys

from pydantic import BaseModel, Field, field_validator, ValidationError
from config import API_ENDPOINTS, COMBINED_COLLECTION_GLOB, COLLECTIONS_DIR, ANALYSIS_BASE

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
    
    def start(self):
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def get_stats(self):
        if self.start_time is None:
            return {}
        
        elapsed = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.memory_start
        
        return {
            "elapsed_time": elapsed,
            "memory_usage_mb": current_memory,
            "memory_delta_mb": memory_delta
        }

# Configure comprehensive logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration with performance monitoring."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('analysis.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

# Initialize logging
logger = setup_logging()

# Data Models (Pydantic)
class AnalysisConfig(BaseModel):
    """Configuration for analysis execution with comprehensive validation."""
    enable_file_source: bool = True
    enable_api_source: bool = True
    enable_diff_analysis: bool = True
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
            
            # Check diff analysis configuration
            if self.enable_diff_analysis and not (self.enable_file_source and self.enable_api_source):
                logger.error("Diff analysis requires both file and API sources")
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

# Data Models (Pydantic) - Complete Implementation
class DataSource(BaseModel):
    """Represents a data source for analysis with comprehensive metadata."""
    source_type: str = Field(..., pattern="^(file|api|dual)$")
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
        return v
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return v

class MessageRecord(BaseModel):
    """Represents a single message record with comprehensive validation."""
    message_id: int = Field(..., ge=1, description="Unique message identifier")
    channel_username: str = Field(..., min_length=1, max_length=100)
    date: datetime = Field(..., description="Message timestamp")
    text: Optional[str] = Field(None, max_length=4096, description="Message text content")
    media_type: Optional[str] = Field(None, pattern="^(text|photo|document|video|audio|voice|sticker|animation|video_note|contact|location|venue|poll|web_page|unsupported)$")
    file_name: Optional[str] = Field(None, max_length=255, description="Original filename")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(None, max_length=100, description="MIME type")
    views: Optional[int] = Field(None, ge=0, description="View count")
    forwards: Optional[int] = Field(None, ge=0, description="Forward count")
    replies: Optional[int] = Field(None, ge=0, description="Reply count")
    reactions: Optional[Dict[str, int]] = Field(None, description="Reaction counts")
    edit_date: Optional[datetime] = Field(None, description="Last edit timestamp")
    reply_to_message_id: Optional[int] = Field(None, ge=1, description="Replied message ID")
    is_forwarded: bool = Field(False, description="Whether message is forwarded")
    is_pinned: bool = Field(False, description="Whether message is pinned")
    is_deleted: bool = Field(False, description="Whether message is deleted")
    
    @field_validator('channel_username')
    @classmethod
    def validate_channel_username(cls, v):
        if not v.startswith('@'):
            raise ValueError("Channel username must start with '@'")
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid date format")
        return v
    
    @field_validator('edit_date')
    @classmethod
    def validate_edit_date(cls, v):
        if v is not None and isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid edit date format")
        return v

class FilenameAnalysisResult(BaseModel):
    """Results from filename analysis with comprehensive metrics."""
    total_files: int = Field(ge=0, description="Total number of files analyzed")
    unique_filenames: int = Field(ge=0, description="Number of unique filenames")
    duplicate_filenames: int = Field(ge=0, description="Number of duplicate filenames")
    duplicate_groups: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Groups of duplicate filenames with counts"
    )
    filename_patterns: Dict[str, int] = Field(
        default_factory=dict, 
        description="Pattern analysis (extensions, lengths, etc.)"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, 
        description="Filename quality metrics"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Recommendations for filename improvements"
    )
    
    @field_validator('duplicate_groups')
    @classmethod
    def validate_duplicate_groups(cls, v):
        for group in v:
            if not isinstance(group, dict) or 'filename' not in group or 'count' not in group:
                raise ValueError("Duplicate groups must have 'filename' and 'count' keys")
        return v

class FilesizeAnalysisResult(BaseModel):
    """Results from filesize analysis with distribution metrics."""
    total_files: int = Field(ge=0, description="Total number of files analyzed")
    total_size_bytes: int = Field(ge=0, description="Total size of all files in bytes")
    unique_sizes: int = Field(ge=0, description="Number of unique file sizes")
    duplicate_sizes: int = Field(ge=0, description="Number of duplicate file sizes")
    duplicate_size_groups: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Groups of files with same size"
    )
    size_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution of files by size bins"
    )
    size_statistics: Dict[str, float] = Field(
        default_factory=dict, 
        description="Statistical measures (mean, median, std, etc.)"
    )
    potential_duplicates: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Files that might be duplicates based on size"
    )
    
    @field_validator('duplicate_size_groups')
    @classmethod
    def validate_duplicate_size_groups(cls, v):
        for group in v:
            if not isinstance(group, dict) or 'size_bytes' not in group or 'count' not in group:
                raise ValueError("Duplicate size groups must have 'size_bytes' and 'count' keys")
        return v

class MessageAnalysisResult(BaseModel):
    """Results from message content analysis with language and pattern detection."""
    total_messages: int = Field(ge=0, description="Total number of messages analyzed")
    text_messages: int = Field(ge=0, description="Number of text messages")
    media_messages: int = Field(ge=0, description="Number of media messages")
    language_distribution: Dict[str, int] = Field(
        default_factory=dict, 
        description="Distribution of messages by detected language"
    )
    content_patterns: Dict[str, int] = Field(
        default_factory=dict, 
        description="Pattern analysis (hashtags, mentions, URLs, etc.)"
    )
    engagement_metrics: Dict[str, float] = Field(
        default_factory=dict, 
        description="Engagement statistics (views, forwards, replies)"
    )
    temporal_patterns: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Time-based patterns (hourly, daily, weekly)"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, 
        description="Content quality metrics"
    )
    
    @field_validator('language_distribution')
    @classmethod
    def validate_language_distribution(cls, v):
        for lang, count in v.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError("Language distribution counts must be non-negative integers")
        return v

class AnalysisResult(BaseModel):
    """Complete analysis result with all analysis types and metadata."""
    analysis_id: str = Field(..., min_length=1, description="Unique analysis identifier")
    timestamp: datetime = Field(..., description="Analysis execution timestamp")
    config: AnalysisConfig = Field(..., description="Configuration used for analysis")
    data_sources: List[DataSource] = Field(
        default_factory=list, 
        description="Data sources used in analysis"
    )
    filename_analysis: Optional[FilenameAnalysisResult] = Field(
        None, 
        description="Filename analysis results"
    )
    filesize_analysis: Optional[FilesizeAnalysisResult] = Field(
        None, 
        description="Filesize analysis results"
    )
    message_analysis: Optional[MessageAnalysisResult] = Field(
        None, 
        description="Message analysis results"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Performance and execution metrics"
    )
    errors: List[str] = Field(
        default_factory=list, 
        description="Any errors encountered during analysis"
    )
    warnings: List[str] = Field(
        default_factory=list, 
        description="Warnings generated during analysis"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of analysis results"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the analysis"
    )
    
    @field_validator('analysis_id')
    @classmethod
    def validate_analysis_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Analysis ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid timestamp format")
        return v

# Base Classes - Data Loading Foundation
class BaseDataLoader:
    """Base class for all data loaders with common functionality."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by converting data types."""
        try:
            # Convert object columns to category if they have low cardinality
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
            
            self.logger.debug(f"Memory optimization completed. Memory usage reduced.")
            return df
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return df
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame structure and content."""
        try:
            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("DataFrame is empty")
                return False
            
            # Check required columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for null values in critical columns
            critical_columns = ['message_id', 'channel_username', 'date']
            for col in critical_columns:
                if col in df.columns and df[col].isnull().any():
                    self.logger.warning(f"Null values found in critical column: {col}")
            
            # Validate data types
            if 'message_id' in df.columns:
                if not pd.api.types.is_integer_dtype(df['message_id']):
                    self.logger.warning("message_id column is not integer type")
            
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    self.logger.warning("date column is not datetime type")
            
            self.logger.debug("DataFrame validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"DataFrame validation failed: {e}")
            return False
    
    def _handle_loading_error(self, error: Exception, context: str) -> None:
        """Handle loading errors with appropriate logging and recovery."""
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg)
        
        # Log additional context for debugging
        if self.config.verbose:
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Handle specific error types
        if isinstance(error, FileNotFoundError):
            self.logger.error("File not found - check file path and permissions")
        elif isinstance(error, pd.errors.EmptyDataError):
            self.logger.error("Data file is empty or corrupted")
        elif isinstance(error, pd.errors.ParserError):
            self.logger.error("Data parsing error - check file format")
        elif isinstance(error, ValidationError):
            self.logger.error("Data validation error - check data structure")
        elif isinstance(error, MemoryError):
            self.logger.error("Memory error - consider reducing chunk size or memory limit")
        else:
            self.logger.error(f"Unexpected error type: {type(error).__name__}")
    
    def _convert_to_message_records(self, df: pd.DataFrame) -> List[MessageRecord]:
        """Convert DataFrame to list of MessageRecord objects."""
        try:
            records = []
            for _, row in df.iterrows():
                try:
                    # Handle date conversion
                    date_value = row.get('date')
                    if pd.isna(date_value):
                        continue
                    
                    if isinstance(date_value, str):
                        date_value = pd.to_datetime(date_value)
                    
                    # Create MessageRecord with proper NaN handling
                    record = MessageRecord(
                        message_id=int(row.get('message_id', 0)),
                        channel_username=str(row.get('channel_username', '')),
                        date=date_value,
                        text=row.get('text') if pd.notna(row.get('text')) else None,
                        media_type=row.get('media_type') if pd.notna(row.get('media_type')) else None,
                        file_name=row.get('file_name') if pd.notna(row.get('file_name')) else None,
                        file_size=int(row.get('file_size')) if pd.notna(row.get('file_size')) else None,
                        mime_type=row.get('mime_type') if pd.notna(row.get('mime_type')) else None,
                        views=int(row.get('views')) if pd.notna(row.get('views')) else None,
                        forwards=int(row.get('forwards')) if pd.notna(row.get('forwards')) else None,
                        replies=int(row.get('replies')) if pd.notna(row.get('replies')) else None,
                        reactions=row.get('reactions') if pd.notna(row.get('reactions')) else None,
                        edit_date=row.get('edit_date') if pd.notna(row.get('edit_date')) else None,
                        reply_to_message_id=int(row.get('reply_to_message_id')) if pd.notna(row.get('reply_to_message_id')) else None,
                        is_forwarded=bool(row.get('is_forwarded', False)),
                        is_pinned=bool(row.get('is_pinned', False)),
                        is_deleted=bool(row.get('is_deleted', False))
                    )
                    records.append(record)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert row to MessageRecord: {e}")
                    continue
            
            self.logger.info(f"Converted {len(records)} records to MessageRecord objects")
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to convert DataFrame to MessageRecord objects: {e}")
            return []
    
    def _create_data_source(self, source_type: str, channel_name: str, 
                           total_records: int, date_range: Tuple[Optional[datetime], Optional[datetime]],
                           quality_score: float = 1.0, metadata: Dict[str, Any] = None) -> DataSource:
        """Create a DataSource object with validation."""
        try:
            return DataSource(
                source_type=source_type,
                channel_name=channel_name,
                total_records=total_records,
                date_range=date_range,
                quality_score=quality_score,
                metadata=metadata or {}
            )
        except Exception as e:
            self.logger.error(f"Failed to create DataSource: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the monitor."""
        return self.performance_monitor.get_stats()
    
    def start_performance_monitoring(self) -> None:
        """Start performance monitoring."""
        self.performance_monitor.start()
    
    def log_performance_stats(self) -> None:
        """Log current performance statistics."""
        stats = self.get_performance_stats()
        if stats:
            self.logger.info(f"Performance stats: {stats}")
        else:
            self.logger.warning("No performance stats available")

# Data Loaders - File-based Data Loading
class FileDataLoader(BaseDataLoader):
    """File-based data loader with chunked processing and memory optimization."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.required_columns = [
            'message_id', 'channel_username', 'date', 'text', 'media_type',
            'file_name', 'file_size', 'mime_type', 'views', 'forwards', 'replies'
        ]
    
    def discover_sources(self) -> List[DataSource]:
        """Discover available data sources from file system."""
        try:
            self.start_performance_monitoring()
            sources = []
            
            # Get collections directory from config
            collections_dir = Path(COLLECTIONS_DIR)
            if not collections_dir.exists():
                self.logger.warning(f"Collections directory not found: {collections_dir}")
                return sources
            
            # Find JSON files matching the pattern
            pattern = COMBINED_COLLECTION_GLOB
            json_files = list(collections_dir.glob(pattern))
            
            if not json_files:
                self.logger.warning(f"No files found matching pattern: {pattern}")
                return sources
            
            self.logger.info(f"Found {len(json_files)} potential data sources")
            
            # Analyze each file to create DataSource objects
            for file_path in json_files:
                try:
                    source = self._analyze_file_source(file_path)
                    if source:
                        sources.append(source)
                        self.logger.debug(f"Added source: {source.channel_name} ({source.total_records} records)")
                except Exception as e:
                    self.logger.warning(f"Failed to analyze file {file_path}: {e}")
                    continue
            
            self.logger.info(f"Successfully discovered {len(sources)} data sources")
            return sources
            
        except Exception as e:
            self._handle_loading_error(e, "discover_sources")
            return []
        finally:
            self.log_performance_stats()
    
    def _analyze_file_source(self, file_path: Path) -> Optional[DataSource]:
        """Analyze a single file to create a DataSource object."""
        try:
            # Quick analysis without loading full data
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to get metadata
                first_line = f.readline()
                if not first_line.strip():
                    return None
                
                # Try to parse as JSON to get structure
                f.seek(0)
                try:
                    # Read a small sample to analyze structure
                    sample_data = f.read(1024)  # Read first 1KB
                    if 'metadata' in sample_data and 'messages' in sample_data:
                        # This looks like our expected format
                        pass
                    else:
                        self.logger.warning(f"Unexpected file format: {file_path}")
                        return None
                except Exception:
                    return None
            
            # Get file stats
            file_stats = file_path.stat()
            file_size_mb = file_size_mb = file_stats.st_size / (1024 * 1024)
            
            # Estimate record count based on file size (rough estimate)
            estimated_records = max(1, int(file_size_mb * 100))  # ~100 records per MB
            
            # Extract channel name from filename
            channel_name = self._extract_channel_name(file_path.name)
            
            # Create metadata
            metadata = {
                'file_path': str(file_path),
                'file_size_mb': round(file_size_mb, 2),
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'estimated_records': estimated_records
            }
            
            # Create DataSource
            return self._create_data_source(
                source_type="file",
                channel_name=channel_name,
                total_records=estimated_records,
                date_range=(None, None),  # Will be updated when data is loaded
                quality_score=0.9,  # Assume good quality for now
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze file source {file_path}: {e}")
            return None
    
    def _extract_channel_name(self, filename: str) -> str:
        """Extract channel name from filename."""
        try:
            # Remove extension
            name_without_ext = filename.replace('.json', '')
            
            # Handle combined files (e.g., tg_books_1_482_combined.json)
            if '_combined' in name_without_ext:
                name_without_ext = name_without_ext.replace('_combined', '')
            
            # Split by underscores and find channel name
            parts = name_without_ext.split('_')
            if len(parts) >= 2:
                # Assume format: tg_channelname_...
                channel_name = parts[1]
                return f"@{channel_name}"
            else:
                return f"@{name_without_ext}"
                
        except Exception:
            return "@unknown_channel"
    
    def load_data(self, source_path: str = None) -> Tuple[pd.DataFrame, DataSource]:
        """Load data from file with chunked processing for large files."""
        try:
            self.start_performance_monitoring()
            
            if source_path:
                file_path = Path(source_path)
            else:
                # Use first available source
                sources = self.discover_sources()
                if not sources:
                    raise FileNotFoundError("No data sources available")
                file_path = Path(sources[0].metadata['file_path'])
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"Loading data from: {file_path}")
            
            # Check file size to determine loading strategy
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 100:  # Large file (>100MB)
                self.logger.info(f"Large file detected ({file_size_mb:.1f}MB), using chunked loading")
                df = self._load_large_file_chunked(file_path)
            else:
                self.logger.info(f"Loading file normally ({file_size_mb:.1f}MB)")
                df = self._load_file_normal(file_path)
            
            # Validate the loaded data
            if not self._validate_dataframe(df, self.required_columns):
                raise ValueError("Loaded data failed validation")
            
            # Optimize memory usage
            df = self._optimize_dataframe_memory(df)
            
            # Create updated DataSource with actual data info
            actual_records = len(df)
            date_range = self._get_date_range(df)
            quality_score = self._calculate_quality_score(df)
            
            source = self._create_data_source(
                source_type="file",
                channel_name=self._extract_channel_name(file_path.name),
                total_records=actual_records,
                date_range=date_range,
                quality_score=quality_score,
                metadata={
                    'file_path': str(file_path),
                    'file_size_mb': round(file_size_mb, 2),
                    'actual_records': actual_records,
                    'loading_method': 'chunked' if file_size_mb > 100 else 'normal'
                }
            )
            
            self.logger.info(f"Successfully loaded {actual_records} records from {file_path.name}")
            return df, source
            
        except Exception as e:
            self._handle_loading_error(e, f"load_data({source_path})")
            return pd.DataFrame(), None
        finally:
            self.log_performance_stats()
    
    def _load_file_normal(self, file_path: Path) -> pd.DataFrame:
        """Load file using standard pandas JSON reading."""
        try:
            # Read JSON file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle nested structure if present
            if isinstance(data, dict) and 'messages' in data:
                messages = data['messages']
            elif isinstance(data, list):
                messages = data
            else:
                raise ValueError("Unexpected JSON structure")
            
            # Convert to DataFrame
            if messages:
                df = pd.DataFrame(messages)
            else:
                df = pd.DataFrame()
            
            # Convert date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load file normally: {e}")
            raise
    
    def _load_large_file_chunked(self, file_path: Path) -> pd.DataFrame:
        """Load large file using chunked processing."""
        try:
            chunks = []
            chunk_size = self.config.chunk_size
            
            self.logger.info(f"Loading file in chunks of {chunk_size} records")
            
            # Read file in chunks
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the file and parse JSON
                data = json.load(f)
                
                # Handle nested structure
                if isinstance(data, dict) and 'messages' in data:
                    messages = data['messages']
                elif isinstance(data, list):
                    messages = data
                else:
                    raise ValueError("Unexpected JSON structure")
                
                # Process in chunks
                for i in range(0, len(messages), chunk_size):
                    chunk_data = messages[i:i + chunk_size]
                    chunk_df = pd.DataFrame(chunk_data)
                    
                    # Convert date column if present
                    if 'date' in chunk_df.columns:
                        chunk_df['date'] = pd.to_datetime(chunk_df['date'], errors='coerce')
                    
                    chunks.append(chunk_df)
                    
                    # Log progress
                    if (i // chunk_size) % 10 == 0:  # Log every 10 chunks
                        self.logger.info(f"Processed {i + len(chunk_data)}/{len(messages)} records")
                    
                    # Memory management
                    if len(chunks) * chunk_size > self.config.memory_limit:
                        self.logger.warning("Memory limit approaching, processing existing chunks")
                        break
            
            # Combine all chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                self.logger.info(f"Combined {len(chunks)} chunks into DataFrame with {len(df)} records")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to load file in chunks: {e}")
            raise
    
    def _get_date_range(self, df: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract date range from DataFrame."""
        try:
            if 'date' in df.columns and not df['date'].isna().all():
                min_date = df['date'].min()
                max_date = df['date'].max()
                return (min_date, max_date)
            return (None, None)
        except Exception as e:
            self.logger.warning(f"Failed to extract date range: {e}")
            return (None, None)
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and validity."""
        try:
            if df.empty:
                return 0.0
            
            total_cells = len(df) * len(df.columns)
            if total_cells == 0:
                return 0.0
            
            # Count non-null values
            non_null_cells = df.count().sum()
            completeness_score = non_null_cells / total_cells
            
            # Check for critical columns
            critical_columns = ['message_id', 'channel_username', 'date']
            critical_completeness = 0.0
            for col in critical_columns:
                if col in df.columns:
                    critical_completeness += (df[col].count() / len(df))
            critical_completeness /= len(critical_columns)
            
            # Combine scores (weight critical columns more)
            quality_score = (completeness_score * 0.6) + (critical_completeness * 0.4)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
            return 0.5  # Default medium quality

# Data Loaders - API-based Data Loading
class ApiDataLoader(BaseDataLoader):
    """API-based data loader with async support and pagination."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.required_columns = [
            'message_id', 'channel_username', 'date', 'text', 'media_type',
            'file_name', 'file_size', 'mime_type', 'views', 'forwards', 'replies'
        ]
        self.session = None
    
    def discover_sources(self) -> List[DataSource]:
        """Discover available data sources from API endpoints."""
        try:
            self.start_performance_monitoring()
            sources = []
            
            # Check if API is available
            if not self._is_endpoint_available():
                self.logger.warning("API endpoint not available")
                return sources
            
            # Get available channels from API
            try:
                channels = self._get_available_channels()
                if not channels:
                    self.logger.warning("No channels available from API")
                    return sources
                
                # Create DataSource for each channel
                for channel in channels:
                    try:
                        source = self._analyze_api_source(channel)
                        if source:
                            sources.append(source)
                            self.logger.debug(f"Added API source: {source.channel_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze API source {channel}: {e}")
                        continue
                
                self.logger.info(f"Successfully discovered {len(sources)} API data sources")
                return sources
                
            except Exception as e:
                self.logger.error(f"Failed to get available channels: {e}")
                return sources
            
        except Exception as e:
            self._handle_loading_error(e, "discover_sources")
            return []
        finally:
            self.log_performance_stats()
    
    def _is_endpoint_available(self) -> bool:
        """Check if API endpoint is available."""
        try:
            import requests
            response = requests.get(
                f"{self.config.api_base_url}/health",
                timeout=self.config.api_timeout
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_available_channels(self) -> List[str]:
        """Get list of available channels from API."""
        try:
            import requests
            response = requests.get(
                f"{self.config.api_base_url}{API_ENDPOINTS['channels']}",
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'channels' in data:
                return data['channels']
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get available channels: {e}")
            return []
    
    def _analyze_api_source(self, channel: str) -> Optional[DataSource]:
        """Analyze a single API source to create a DataSource object."""
        try:
            # Get basic stats for the channel
            stats = self._get_channel_stats(channel)
            if not stats:
                return None
            
            # Create metadata
            metadata = {
                'api_endpoint': f"{self.config.api_base_url}{API_ENDPOINTS['messages']}",
                'channel': channel,
                'total_messages': stats.get('total_messages', 0),
                'date_range': stats.get('date_range', {}),
                'last_updated': stats.get('last_updated', None)
            }
            
            # Create DataSource
            return self._create_data_source(
                source_type="api",
                channel_name=channel,
                total_records=stats.get('total_messages', 0),
                date_range=(None, None),  # Will be updated when data is loaded
                quality_score=0.95,  # Assume good quality for API data
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze API source {channel}: {e}")
            return None
    
    def _get_channel_stats(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific channel."""
        try:
            import requests
            response = requests.get(
                f"{self.config.api_base_url}{API_ENDPOINTS['stats']}",
                params={'channel': channel},
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.warning(f"Failed to get stats for channel {channel}: {e}")
            return None
    
    def load_data(self, source_path: str = None) -> Tuple[pd.DataFrame, DataSource]:
        """Load data from API with pagination and async support."""
        try:
            self.start_performance_monitoring()
            
            if source_path:
                channel = source_path
            else:
                # Use first available source
                sources = self.discover_sources()
                if not sources:
                    raise ValueError("No API data sources available")
                channel = sources[0].channel_name
            
            self.logger.info(f"Loading data from API for channel: {channel}")
            
            # Load data asynchronously
            df = asyncio.run(self.load_data_async(channel))
            
            if df.empty:
                self.logger.warning(f"No data loaded for channel: {channel}")
                return df, None
            
            # Validate the loaded data
            if not self._validate_dataframe(df, self.required_columns):
                raise ValueError("Loaded data failed validation")
            
            # Optimize memory usage
            df = self._optimize_dataframe_memory(df)
            
            # Create updated DataSource with actual data info
            actual_records = len(df)
            date_range = self._get_date_range(df)
            quality_score = self._calculate_quality_score(df)
            
            source = self._create_data_source(
                source_type="api",
                channel_name=channel,
                total_records=actual_records,
                date_range=date_range,
                quality_score=quality_score,
                metadata={
                    'api_endpoint': f"{self.config.api_base_url}{API_ENDPOINTS['messages']}",
                    'channel': channel,
                    'actual_records': actual_records,
                    'loading_method': 'async_pagination'
                }
            )
            
            self.logger.info(f"Successfully loaded {actual_records} records from API for {channel}")
            return df, source
            
        except Exception as e:
            self._handle_loading_error(e, f"load_data({source_path})")
            return pd.DataFrame(), None
        finally:
            self.log_performance_stats()
    
    async def load_data_async(self, channel: str) -> pd.DataFrame:
        """Load data from API asynchronously with pagination."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.api_timeout)) as session:
                self.session = session
                
                # Get total count first
                total_count = await self._get_total_count(channel)
                if total_count == 0:
                    self.logger.warning(f"No data available for channel: {channel}")
                    return pd.DataFrame()
                
                self.logger.info(f"Loading {total_count} records from API for {channel}")
                
                # Load data in pages
                all_data = []
                page = 1
                items_per_page = self.config.items_per_page
                
                while True:
                    try:
                        page_data = await self._fetch_page_with_retry(
                            channel, page, items_per_page
                        )
                        
                        if not page_data:
                            break
                        
                        all_data.extend(page_data)
                        
                        # Log progress
                        if page % 10 == 0:  # Log every 10 pages
                            self.logger.info(f"Loaded {len(all_data)}/{total_count} records")
                        
                        # Check if we've loaded all data
                        if len(page_data) < items_per_page:
                            break
                        
                        page += 1
                        
                        # Memory management
                        if len(all_data) > self.config.memory_limit:
                            self.logger.warning("Memory limit reached, stopping data loading")
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Failed to fetch page {page}: {e}")
                        break
                
                # Convert to DataFrame
                if all_data:
                    df = pd.DataFrame(all_data)
                    
                    # Convert date column if present
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    self.logger.info(f"Successfully loaded {len(df)} records from API")
                    return df
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Failed to load data asynchronously: {e}")
            return pd.DataFrame()
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    async def _get_total_count(self, channel: str) -> int:
        """Get total count of records for a channel."""
        try:
            url = f"{self.config.api_base_url}{API_ENDPOINTS['stats']}"
            params = {'channel': channel}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('total_messages', 0)
                else:
                    self.logger.warning(f"Failed to get total count: {response.status}")
                    return 0
                    
        except Exception as e:
            self.logger.error(f"Failed to get total count: {e}")
            return 0
    
    async def _fetch_page_with_retry(self, channel: str, page: int, items_per_page: int) -> List[Dict[str, Any]]:
        """Fetch a page of data with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                return await self._fetch_page(channel, page, items_per_page)
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"All attempts failed for page {page}: {e}")
                    raise
        
        return []
    
    async def _fetch_page(self, channel: str, page: int, items_per_page: int) -> List[Dict[str, Any]]:
        """Fetch a single page of data from API."""
        try:
            url = f"{self.config.api_base_url}{API_ENDPOINTS['messages']}"
            params = {
                'channel': channel,
                'page': page,
                'per_page': items_per_page
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        if 'messages' in data:
                            return data['messages']
                        elif 'data' in data:
                            return data['data']
                        else:
                            return []
                    else:
                        return []
                        
                elif response.status == 404:
                    self.logger.warning(f"Channel not found: {channel}")
                    return []
                else:
                    self.logger.error(f"API request failed: {response.status}")
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch page {page}: {e}")
            raise
    
    def _get_date_range(self, df: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract date range from DataFrame."""
        try:
            if 'date' in df.columns and not df['date'].isna().all():
                min_date = df['date'].min()
                max_date = df['date'].max()
                return (min_date, max_date)
            return (None, None)
        except Exception as e:
            self.logger.warning(f"Failed to extract date range: {e}")
            return (None, None)
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and validity."""
        try:
            if df.empty:
                return 0.0
            
            total_cells = len(df) * len(df.columns)
            if total_cells == 0:
                return 0.0
            
            # Count non-null values
            non_null_cells = df.count().sum()
            completeness_score = non_null_cells / total_cells
            
            # Check for critical columns
            critical_columns = ['message_id', 'channel_username', 'date']
            critical_completeness = 0.0
            for col in critical_columns:
                if col in df.columns:
                    critical_completeness += (df[col].count() / len(df))
            critical_completeness /= len(critical_columns)
            
            # Combine scores (weight critical columns more)
            quality_score = (completeness_score * 0.6) + (critical_completeness * 0.4)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
            return 0.5  # Default medium quality

# Analysis Engines - Core Analysis Functionality
class FilenameAnalyzer:
    """Analyzes filename patterns, duplicates, and quality metrics."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
    
    def analyze(self, df: pd.DataFrame) -> FilenameAnalysisResult:
        """Perform comprehensive filename analysis."""
        try:
            self.performance_monitor.start()
            
            if df.empty:
                self.logger.warning("Empty DataFrame provided for filename analysis")
                return FilenameAnalysisResult(
                    total_files=0,
                    unique_filenames=0,
                    duplicate_filenames=0,
                    duplicate_groups=[],
                    filename_patterns={},
                    quality_metrics={},
                    recommendations=[]
                )
            
            # Filter for files with filenames
            file_df = df[df['file_name'].notna() & (df['file_name'] != '')].copy()
            
            if file_df.empty:
                self.logger.warning("No files with filenames found for analysis")
                return FilenameAnalysisResult(
                    total_files=0,
                    unique_filenames=0,
                    duplicate_filenames=0,
                    duplicate_groups=[],
                    filename_patterns={},
                    quality_metrics={},
                    recommendations=[]
                )
            
            self.logger.info(f"Analyzing {len(file_df)} files with filenames")
            
            # Perform analysis
            total_files = len(file_df)
            unique_filenames = file_df['file_name'].nunique()
            duplicate_groups = self._find_duplicate_filenames(file_df)
            duplicate_filenames = sum(group['count'] for group in duplicate_groups if group['count'] > 1)
            filename_patterns = self._analyze_filename_patterns(file_df)
            quality_metrics = self._calculate_filename_quality_metrics(file_df)
            recommendations = self._generate_filename_recommendations(file_df, quality_metrics)
            
            result = FilenameAnalysisResult(
                total_files=total_files,
                unique_filenames=unique_filenames,
                duplicate_filenames=duplicate_filenames,
                duplicate_groups=duplicate_groups,
                filename_patterns=filename_patterns,
                quality_metrics=quality_metrics,
                recommendations=recommendations
            )
            
            self.logger.info(f"Filename analysis completed: {total_files} files, {unique_filenames} unique, {duplicate_filenames} duplicates")
            return result
            
        except Exception as e:
            self.logger.error(f"Filename analysis failed: {e}")
            raise
        finally:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.logger.info(f"Performance stats: {stats}")
    
    def _find_duplicate_filenames(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find duplicate filenames and group them."""
        try:
            filename_counts = df['file_name'].value_counts()
            duplicate_groups = []
            
            for filename, count in filename_counts.items():
                if count > 1:
                    # Get additional info about duplicates
                    duplicate_files = df[df['file_name'] == filename]
                    file_sizes = duplicate_files['file_size'].dropna().tolist()
                    channels = duplicate_files['channel_username'].unique().tolist()
                    
                    group_info = {
                        'filename': filename,
                        'count': int(count),
                        'file_sizes': file_sizes,
                        'channels': channels,
                        'size_variation': len(set(file_sizes)) > 1 if file_sizes else False
                    }
                    duplicate_groups.append(group_info)
            
            # Sort by count (descending)
            duplicate_groups.sort(key=lambda x: x['count'], reverse=True)
            return duplicate_groups
            
        except Exception as e:
            self.logger.error(f"Failed to find duplicate filenames: {e}")
            return []
    
    def _analyze_filename_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze filename patterns and extensions."""
        try:
            patterns = {}
            
            # Filename lengths
            lengths = df['file_name'].str.len()
            patterns['avg_length'] = int(lengths.mean())
            patterns['min_length'] = int(lengths.min())
            patterns['max_length'] = int(lengths.max())
            
            # Special characters
            special_chars = df['file_name'].str.count(r'[^a-zA-Z0-9._-]')
            patterns['files_with_special_chars'] = int((special_chars > 0).sum())
            patterns['avg_special_chars'] = int(special_chars.mean())
            
            # Total files with extensions
            extensions = df['file_name'].str.extract(r'\.([^.]+)$')[0].dropna()
            patterns['files_with_extensions'] = len(extensions)
            
            # Total files with prefixes
            prefixes = df['file_name'].str.extract(r'^([a-zA-Z]+)')[0].dropna()
            patterns['files_with_prefixes'] = len(prefixes)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze filename patterns: {e}")
            return {}
    
    def _calculate_filename_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate filename quality metrics."""
        try:
            metrics = {}
            
            # Completeness
            total_files = len(df)
            files_with_names = df['file_name'].notna().sum()
            metrics['completeness'] = files_with_names / total_files if total_files > 0 else 0.0
            
            # Uniqueness
            unique_names = df['file_name'].nunique()
            metrics['uniqueness'] = unique_names / total_files if total_files > 0 else 0.0
            
            # Length quality (optimal range: 10-50 characters)
            lengths = df['file_name'].str.len()
            optimal_length = ((lengths >= 10) & (lengths <= 50)).sum()
            metrics['length_quality'] = optimal_length / total_files if total_files > 0 else 0.0
            
            # Special character usage (lower is better)
            special_chars = df['file_name'].str.count(r'[^a-zA-Z0-9._-]')
            clean_names = (special_chars == 0).sum()
            metrics['cleanliness'] = clean_names / total_files if total_files > 0 else 0.0
            
            # Descriptive quality (has meaningful parts)
            has_meaningful_parts = df['file_name'].str.contains(r'[a-zA-Z]{3,}', na=False).sum()
            metrics['descriptiveness'] = has_meaningful_parts / total_files if total_files > 0 else 0.0
            
            # Overall quality score
            metrics['overall_quality'] = (
                metrics['completeness'] * 0.2 +
                metrics['uniqueness'] * 0.3 +
                metrics['length_quality'] * 0.2 +
                metrics['cleanliness'] * 0.15 +
                metrics['descriptiveness'] * 0.15
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate filename quality metrics: {e}")
            return {}
    
    def _generate_filename_recommendations(self, df: pd.DataFrame, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for filename improvements."""
        try:
            recommendations = []
            
            # Check completeness
            if quality_metrics.get('completeness', 0) < 0.95:
                recommendations.append("Some files are missing filenames - ensure all files have proper names")
            
            # Check uniqueness
            if quality_metrics.get('uniqueness', 0) < 0.8:
                recommendations.append("High number of duplicate filenames detected - consider adding timestamps or unique identifiers")
            
            # Check length quality
            if quality_metrics.get('length_quality', 0) < 0.7:
                recommendations.append("Many filenames are too short or too long - aim for 10-50 characters")
            
            # Check cleanliness
            if quality_metrics.get('cleanliness', 0) < 0.8:
                recommendations.append("Some filenames contain special characters - use only alphanumeric characters, dots, hyphens, and underscores")
            
            # Check descriptiveness
            if quality_metrics.get('descriptiveness', 0) < 0.7:
                recommendations.append("Some filenames lack descriptive content - include meaningful words to describe the file content")
            
            # Overall quality
            if quality_metrics.get('overall_quality', 0) < 0.6:
                recommendations.append("Overall filename quality is low - consider implementing a filename standardization policy")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate filename recommendations: {e}")
            return []


class FilesizeAnalyzer:
    """Analyzes file sizes, distributions, and potential duplicates."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
    
    def analyze(self, df: pd.DataFrame) -> FilesizeAnalysisResult:
        """Perform comprehensive filesize analysis."""
        try:
            self.performance_monitor.start()
            
            if df.empty:
                self.logger.warning("Empty DataFrame provided for filesize analysis")
                return FilesizeAnalysisResult(
                    total_files=0,
                    total_size_bytes=0,
                    unique_sizes=0,
                    duplicate_sizes=0,
                    duplicate_size_groups=[],
                    size_distribution={},
                    size_statistics={},
                    potential_duplicates=[]
                )
            
            # Filter for files with sizes
            file_df = df[df['file_size'].notna() & (df['file_size'] > 0)].copy()
            
            if file_df.empty:
                self.logger.warning("No files with sizes found for analysis")
                return FilesizeAnalysisResult(
                    total_files=0,
                    total_size_bytes=0,
                    unique_sizes=0,
                    duplicate_sizes=0,
                    duplicate_size_groups=[],
                    size_distribution={},
                    size_statistics={},
                    potential_duplicates=[]
                )
            
            self.logger.info(f"Analyzing {len(file_df)} files with sizes")
            
            # Perform analysis
            total_files = len(file_df)
            total_size_bytes = int(file_df['file_size'].sum())
            unique_sizes = file_df['file_size'].nunique()
            duplicate_size_groups = self._find_duplicate_sizes(file_df)
            duplicate_sizes = sum(group['count'] for group in duplicate_size_groups if group['count'] > 1)
            size_distribution = self._analyze_size_distribution(file_df)
            size_statistics = self._calculate_size_statistics(file_df)
            potential_duplicates = self._find_potential_duplicates(file_df)
            
            result = FilesizeAnalysisResult(
                total_files=total_files,
                total_size_bytes=total_size_bytes,
                unique_sizes=unique_sizes,
                duplicate_sizes=duplicate_sizes,
                duplicate_size_groups=duplicate_size_groups,
                size_distribution=size_distribution,
                size_statistics=size_statistics,
                potential_duplicates=potential_duplicates
            )
            
            self.logger.info(f"Filesize analysis completed: {total_files} files, {total_size_bytes} bytes total, {unique_sizes} unique sizes")
            return result
            
        except Exception as e:
            self.logger.error(f"Filesize analysis failed: {e}")
            raise
        finally:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.logger.info(f"Performance stats: {stats}")
    
    def _find_duplicate_sizes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find files with duplicate sizes."""
        try:
            size_counts = df['file_size'].value_counts()
            duplicate_groups = []
            
            for size, count in size_counts.items():
                if count > 1:
                    # Get additional info about duplicates
                    duplicate_files = df[df['file_size'] == size]
                    filenames = duplicate_files['file_name'].dropna().tolist()
                    channels = duplicate_files['channel_username'].unique().tolist()
                    
                    group_info = {
                        'size_bytes': int(size),
                        'count': int(count),
                        'filenames': filenames,
                        'channels': channels,
                        'size_mb': round(size / (1024 * 1024), 2)
                    }
                    duplicate_groups.append(group_info)
            
            # Sort by count (descending)
            duplicate_groups.sort(key=lambda x: x['count'], reverse=True)
            return duplicate_groups
            
        except Exception as e:
            self.logger.error(f"Failed to find duplicate sizes: {e}")
            return []
    
    def _analyze_size_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze file size distribution in bins."""
        try:
            distribution = {}
            
            # Define size bins
            size_bins = {
                'tiny': (0, 1024),  # < 1KB
                'small': (1024, 1024 * 1024),  # 1KB - 1MB
                'medium': (1024 * 1024, 10 * 1024 * 1024),  # 1MB - 10MB
                'large': (10 * 1024 * 1024, 100 * 1024 * 1024),  # 10MB - 100MB
                'huge': (100 * 1024 * 1024, float('inf'))  # > 100MB
            }
            
            for bin_name, (min_size, max_size) in size_bins.items():
                count = ((df['file_size'] >= min_size) & (df['file_size'] < max_size)).sum()
                distribution[bin_name] = int(count)
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Failed to analyze size distribution: {e}")
            return {}
    
    def _calculate_size_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical measures for file sizes."""
        try:
            sizes = df['file_size']
            statistics = {}
            
            # Basic statistics
            statistics['mean'] = float(sizes.mean())
            statistics['median'] = float(sizes.median())
            statistics['std'] = float(sizes.std())
            statistics['min'] = float(sizes.min())
            statistics['max'] = float(sizes.max())
            
            # Percentiles
            statistics['p25'] = float(sizes.quantile(0.25))
            statistics['p75'] = float(sizes.quantile(0.75))
            statistics['p90'] = float(sizes.quantile(0.90))
            statistics['p95'] = float(sizes.quantile(0.95))
            statistics['p99'] = float(sizes.quantile(0.99))
            
            # Size in MB for readability
            statistics['mean_mb'] = round(statistics['mean'] / (1024 * 1024), 2)
            statistics['median_mb'] = round(statistics['median'] / (1024 * 1024), 2)
            statistics['max_mb'] = round(statistics['max'] / (1024 * 1024), 2)
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate size statistics: {e}")
            return {}
    
    def _find_potential_duplicates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find files that might be duplicates based on size and other factors."""
        try:
            potential_duplicates = []
            
            # Group by size and find groups with multiple files
            size_groups = df.groupby('file_size')
            
            for size, group in size_groups:
                if len(group) > 1:
                    # Check if files have similar names or are from same channel
                    files_info = []
                    for _, file_row in group.iterrows():
                        file_info = {
                            'message_id': int(file_row['message_id']),
                            'filename': file_row.get('file_name', ''),
                            'channel': file_row.get('channel_username', ''),
                            'date': file_row.get('date', ''),
                            'mime_type': file_row.get('mime_type', '')
                        }
                        files_info.append(file_info)
                    
                    # Calculate similarity score
                    similarity_score = self._calculate_similarity_score(files_info)
                    
                    if similarity_score > 0.3:  # Threshold for potential duplicates
                        potential_duplicates.append({
                            'size_bytes': int(size),
                            'size_mb': round(size / (1024 * 1024), 2),
                            'file_count': len(files_info),
                            'files': files_info,
                            'similarity_score': similarity_score
                        })
            
            # Sort by similarity score (descending)
            potential_duplicates.sort(key=lambda x: x['similarity_score'], reverse=True)
            return potential_duplicates[:20]  # Return top 20 potential duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to find potential duplicates: {e}")
            return []
    
    def _calculate_similarity_score(self, files_info: List[Dict[str, Any]]) -> float:
        """Calculate similarity score for files with same size."""
        try:
            if len(files_info) < 2:
                return 0.0
            
            score = 0.0
            total_comparisons = 0
            
            # Compare each pair of files
            for i in range(len(files_info)):
                for j in range(i + 1, len(files_info)):
                    file1, file2 = files_info[i], files_info[j]
                    pair_score = 0.0
                    
                    # Same channel
                    if file1['channel'] == file2['channel']:
                        pair_score += 0.3
                    
                    # Similar filenames
                    if file1['filename'] and file2['filename']:
                        filename_similarity = self._calculate_filename_similarity(
                            file1['filename'], file2['filename']
                        )
                        pair_score += filename_similarity * 0.4
                    
                    # Same MIME type
                    if file1['mime_type'] == file2['mime_type'] and file1['mime_type']:
                        pair_score += 0.3
                    
                    score += pair_score
                    total_comparisons += 1
            
            return score / total_comparisons if total_comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity score: {e}")
            return 0.0
    
    def _calculate_filename_similarity(self, filename1: str, filename2: str) -> float:
        """Calculate similarity between two filenames."""
        try:
            if not filename1 or not filename2:
                return 0.0
            
            # Simple similarity based on common characters
            set1 = set(filename1.lower())
            set2 = set(filename2.lower())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate filename similarity: {e}")
            return 0.0


class MessageAnalyzer:
    """Analyzes message content, patterns, and language detection."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
    
    def analyze(self, df: pd.DataFrame) -> MessageAnalysisResult:
        """Perform comprehensive message analysis."""
        try:
            self.performance_monitor.start()
            
            if df.empty:
                self.logger.warning("Empty DataFrame provided for message analysis")
                return MessageAnalysisResult(
                    total_messages=0,
                    text_messages=0,
                    media_messages=0,
                    language_distribution={},
                    content_patterns={},
                    engagement_metrics={},
                    temporal_patterns={},
                    quality_metrics={}
                )
            
            self.logger.info(f"Analyzing {len(df)} messages")
            
            # Perform analysis
            total_messages = len(df)
            text_messages = len(df[df['media_type'] == 'text'])
            media_messages = len(df[df['media_type'] != 'text'])
            language_distribution = self._analyze_language_distribution(df)
            content_patterns = self._analyze_content_patterns(df)
            engagement_metrics = self._calculate_engagement_metrics(df)
            temporal_patterns = self._analyze_temporal_patterns(df)
            quality_metrics = self._calculate_message_quality_metrics(df)
            
            result = MessageAnalysisResult(
                total_messages=total_messages,
                text_messages=text_messages,
                media_messages=media_messages,
                language_distribution=language_distribution,
                content_patterns=content_patterns,
                engagement_metrics=engagement_metrics,
                temporal_patterns=temporal_patterns,
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Message analysis completed: {total_messages} messages, {text_messages} text, {media_messages} media")
            return result
            
        except Exception as e:
            self.logger.error(f"Message analysis failed: {e}")
            raise
        finally:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.logger.info(f"Performance stats: {stats}")
    
    def _analyze_language_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze language distribution in messages."""
        try:
            # Filter for text messages
            text_df = df[df['text'].notna() & (df['text'] != '')].copy()
            
            if text_df.empty:
                return {}
            
            # Simple language detection based on character frequency
            language_counts = {}
            
            # Process in chunks for memory efficiency
            chunk_size = min(1000, len(text_df))
            for i in range(0, len(text_df), chunk_size):
                chunk = text_df.iloc[i:i + chunk_size]
                chunk_languages = self._detect_languages_chunk(chunk['text'])
                
                for lang, count in chunk_languages.items():
                    language_counts[lang] = language_counts.get(lang, 0) + count
            
            return language_counts
            
        except Exception as e:
            self.logger.error(f"Failed to analyze language distribution: {e}")
            return {}
    
    def _detect_languages_chunk(self, texts: pd.Series) -> Dict[str, int]:
        """Detect languages in a chunk of texts."""
        try:
            language_counts = {}
            
            for text in texts:
                if pd.isna(text) or not text:
                    continue
                
                # Simple language detection based on character frequency
                language = self._detect_language_simple(text)
                language_counts[language] = language_counts.get(language, 0) + 1
            
            return language_counts
            
        except Exception as e:
            self.logger.error(f"Failed to detect languages in chunk: {e}")
            return {}
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on character frequency."""
        try:
            if not text or len(text) < 10:
                return 'unknown'
            
            # Count different character types
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            
            total_chars = len([c for c in text if c.isalpha()])
            
            if total_chars == 0:
                return 'unknown'
            
            # Determine language based on character distribution
            if cyrillic_chars / total_chars > 0.3:
                return 'russian'
            elif arabic_chars / total_chars > 0.3:
                return 'arabic'
            elif chinese_chars / total_chars > 0.3:
                return 'chinese'
            elif latin_chars / total_chars > 0.7:
                return 'english'
            else:
                return 'mixed'
                
        except Exception as e:
            self.logger.error(f"Failed to detect language: {e}")
            return 'unknown'
    
    def _analyze_content_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze content patterns in messages."""
        try:
            patterns = {}
            
            # Filter for text messages
            text_df = df[df['text'].notna() & (df['text'] != '')].copy()
            
            if text_df.empty:
                return patterns
            
            # Hashtags
            hashtags = text_df['text'].str.count(r'#\w+').sum()
            patterns['hashtags'] = int(hashtags)
            
            # Mentions
            mentions = text_df['text'].str.count(r'@\w+').sum()
            patterns['mentions'] = int(mentions)
            
            # URLs
            urls = text_df['text'].str.count(r'https?://\S+').sum()
            patterns['urls'] = int(urls)
            
            # Emojis (simple detection)
            emojis = text_df['text'].str.count(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]').sum()
            patterns['emojis'] = int(emojis)
            
            # Questions
            questions = text_df['text'].str.count(r'\?').sum()
            patterns['questions'] = int(questions)
            
            # Exclamations
            exclamations = text_df['text'].str.count(r'!').sum()
            patterns['exclamations'] = int(exclamations)
            
            # Average message length
            avg_length = text_df['text'].str.len().mean()
            patterns['avg_message_length'] = int(round(avg_length))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze content patterns: {e}")
            return {}
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate engagement metrics for messages."""
        try:
            metrics = {}
            
            # Views
            views = df['views'].dropna()
            if not views.empty:
                metrics['avg_views'] = float(views.mean())
                metrics['median_views'] = float(views.median())
                metrics['max_views'] = float(views.max())
            
            # Forwards
            forwards = df['forwards'].dropna()
            if not forwards.empty:
                metrics['avg_forwards'] = float(forwards.mean())
                metrics['median_forwards'] = float(forwards.median())
                metrics['max_forwards'] = float(forwards.max())
            
            # Replies
            replies = df['replies'].dropna()
            if not replies.empty:
                metrics['avg_replies'] = float(replies.mean())
                metrics['median_replies'] = float(replies.median())
                metrics['max_replies'] = float(replies.max())
            
            # Engagement rate (forwards + replies) / views
            if not views.empty and not (forwards.empty and replies.empty):
                total_engagement = (forwards.fillna(0) + replies.fillna(0)).sum()
                total_views = views.sum()
                if total_views > 0:
                    metrics['engagement_rate'] = float(total_engagement / total_views)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate engagement metrics: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in messages."""
        try:
            patterns = {}
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df_with_dates = df[df['date'].notna()].copy()
                if not df_with_dates.empty:
                    df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
                    
                    # Hourly patterns
                    df_with_dates['hour'] = df_with_dates['date'].dt.hour
                    hourly_counts = df_with_dates['hour'].value_counts().sort_index()
                    patterns['hourly'] = hourly_counts.to_dict()
                    
                    # Daily patterns
                    df_with_dates['day_of_week'] = df_with_dates['date'].dt.day_name()
                    daily_counts = df_with_dates['day_of_week'].value_counts()
                    patterns['daily'] = daily_counts.to_dict()
                    
                    # Monthly patterns
                    df_with_dates['month'] = df_with_dates['date'].dt.month
                    monthly_counts = df_with_dates['month'].value_counts().sort_index()
                    patterns['monthly'] = monthly_counts.to_dict()
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze temporal patterns: {e}")
            return {}
    
    def _calculate_message_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate message quality metrics."""
        try:
            metrics = {}
            
            # Completeness
            total_messages = len(df)
            messages_with_text = df['text'].notna().sum()
            metrics['text_completeness'] = messages_with_text / total_messages if total_messages > 0 else 0.0
            
            # Media completeness
            media_messages = df[df['media_type'] != 'text']
            media_with_files = media_messages['file_name'].notna().sum()
            metrics['media_completeness'] = media_with_files / len(media_messages) if len(media_messages) > 0 else 0.0
            
            # Engagement completeness
            messages_with_views = df['views'].notna().sum()
            metrics['engagement_completeness'] = messages_with_views / total_messages if total_messages > 0 else 0.0
            
            # Content quality (based on text length and patterns)
            text_df = df[df['text'].notna() & (df['text'] != '')]
            if not text_df.empty:
                avg_length = text_df['text'].str.len().mean()
                metrics['avg_content_length'] = float(avg_length)
                
                # Quality score based on length (optimal: 50-500 characters)
                optimal_length = ((text_df['text'].str.len() >= 50) & (text_df['text'].str.len() <= 500)).sum()
                metrics['content_quality'] = optimal_length / len(text_df)
            else:
                metrics['avg_content_length'] = 0.0
                metrics['content_quality'] = 0.0
            
            # Overall quality
            metrics['overall_quality'] = (
                metrics['text_completeness'] * 0.3 +
                metrics['media_completeness'] * 0.2 +
                metrics['engagement_completeness'] * 0.2 +
                metrics['content_quality'] * 0.3
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate message quality metrics: {e}")
            return {}

# Output Management - JSON Output with Pandas
class JsonOutputManager:
    """Manages JSON output generation using pandas for all JSON operations."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_monitor = PerformanceMonitor()
    
    def generate_analysis_report(self, 
                               filename_result: FilenameAnalysisResult,
                               filesize_result: FilesizeAnalysisResult,
                               message_result: MessageAnalysisResult,
                               data_sources: List[DataSource],
                               output_path: str = None) -> str:
        """Generate comprehensive analysis report in JSON format."""
        try:
            self.performance_monitor.start()
            
            # Create output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"analysis_report_{timestamp}.json"
            
            self.logger.info(f"Generating analysis report: {output_path}")
            
            # Create comprehensive analysis result
            analysis_result = AnalysisResult(
                analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                config=self.config,
                data_sources=data_sources,
                filename_analysis=filename_result,
                filesize_analysis=filesize_result,
                message_analysis=message_result,
                summary=self._generate_summary(filename_result, filesize_result, message_result),
                metadata=self._generate_metadata()
            )
            
            # Convert to dictionary for JSON serialization
            report_data = self._prepare_report_data(analysis_result)
            
            # Write JSON using pandas
            self._write_json_with_pandas(report_data, output_path)
            
            self.logger.info(f"Analysis report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {e}")
            raise
        finally:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.logger.info(f"Performance stats: {stats}")
    
    def generate_individual_reports(self,
                                  filename_result: FilenameAnalysisResult,
                                  filesize_result: FilesizeAnalysisResult,
                                  message_result: MessageAnalysisResult,
                                  output_dir: str = None) -> Dict[str, str]:
        """Generate individual analysis reports for each analysis type."""
        try:
            self.performance_monitor.start()
            
            # Create output directory if not provided
            if not output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"analysis_output_{timestamp}"
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Generating individual reports in: {output_dir}")
            
            report_paths = {}
            
            # Generate filename analysis report
            filename_path = Path(output_dir) / "filename_analysis.json"
            filename_data = self._prepare_filename_report_data(filename_result)
            self._write_json_with_pandas(filename_data, str(filename_path))
            report_paths['filename'] = str(filename_path)
            
            # Generate filesize analysis report
            filesize_path = Path(output_dir) / "filesize_analysis.json"
            filesize_data = self._prepare_filesize_report_data(filesize_result)
            self._write_json_with_pandas(filesize_data, str(filesize_path))
            report_paths['filesize'] = str(filesize_path)
            
            # Generate message analysis report
            message_path = Path(output_dir) / "message_analysis.json"
            message_data = self._prepare_message_report_data(message_result)
            self._write_json_with_pandas(message_data, str(message_path))
            report_paths['message'] = str(message_path)
            
            # Generate summary report
            summary_path = Path(output_dir) / "analysis_summary.json"
            summary_data = self._prepare_summary_report_data(filename_result, filesize_result, message_result)
            self._write_json_with_pandas(summary_data, str(summary_path))
            report_paths['summary'] = str(summary_path)
            
            self.logger.info(f"Individual reports generated successfully in: {output_dir}")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Failed to generate individual reports: {e}")
            raise
        finally:
            stats = self.performance_monitor.get_stats()
            if stats:
                self.logger.info(f"Performance stats: {stats}")
    
    def _prepare_report_data(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Prepare analysis result data for JSON serialization."""
        try:
            # Convert Pydantic model to dictionary
            report_data = analysis_result.model_dump()
            
            # Convert datetime objects to ISO strings
            if 'timestamp' in report_data:
                report_data['timestamp'] = report_data['timestamp'].isoformat()
            
            # Convert data sources
            if 'data_sources' in report_data:
                for source in report_data['data_sources']:
                    if 'date_range' in source and source['date_range']:
                        # Convert tuple to list for modification
                        date_range = list(source['date_range'])
                        if date_range[0]:
                            date_range[0] = date_range[0].isoformat()
                        if date_range[1]:
                            date_range[1] = date_range[1].isoformat()
                        source['date_range'] = date_range
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare report data: {e}")
            raise
    
    def _prepare_filename_report_data(self, result: FilenameAnalysisResult) -> Dict[str, Any]:
        """Prepare filename analysis data for JSON serialization."""
        try:
            data = result.model_dump()
            
            # Add additional metadata
            data['report_type'] = 'filename_analysis'
            data['generated_at'] = datetime.now().isoformat()
            data['analysis_version'] = '1.0'
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare filename report data: {e}")
            raise
    
    def _prepare_filesize_report_data(self, result: FilesizeAnalysisResult) -> Dict[str, Any]:
        """Prepare filesize analysis data for JSON serialization."""
        try:
            data = result.model_dump()
            
            # Add additional metadata
            data['report_type'] = 'filesize_analysis'
            data['generated_at'] = datetime.now().isoformat()
            data['analysis_version'] = '1.0'
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare filesize report data: {e}")
            raise
    
    def _prepare_message_report_data(self, result: MessageAnalysisResult) -> Dict[str, Any]:
        """Prepare message analysis data for JSON serialization."""
        try:
            data = result.model_dump()
            
            # Add additional metadata
            data['report_type'] = 'message_analysis'
            data['generated_at'] = datetime.now().isoformat()
            data['analysis_version'] = '1.0'
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare message report data: {e}")
            raise
    
    def _prepare_summary_report_data(self, 
                                   filename_result: FilenameAnalysisResult,
                                   filesize_result: FilesizeAnalysisResult,
                                   message_result: MessageAnalysisResult) -> Dict[str, Any]:
        """Prepare summary report data for JSON serialization."""
        try:
            summary_data = {
                'report_type': 'analysis_summary',
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '1.0',
                'summary': self._generate_summary(filename_result, filesize_result, message_result),
                'key_metrics': self._generate_key_metrics(filename_result, filesize_result, message_result),
                'recommendations': self._generate_recommendations(filename_result, filesize_result, message_result),
                'metadata': self._generate_metadata()
            }
            
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare summary report data: {e}")
            raise
    
    def _write_json_with_pandas(self, data: Dict[str, Any], output_path: str) -> None:
        """Write JSON data using pandas for consistency."""
        try:
            # Convert data to DataFrame for pandas JSON operations
            # Create a single-row DataFrame with the data
            df = pd.DataFrame([data])
            
            # Use pandas to write JSON
            df.to_json(output_path, orient='records', indent=2, date_format='iso')
            
            self.logger.debug(f"JSON written successfully: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write JSON with pandas: {e}")
            raise
    
    def _generate_summary(self, 
                         filename_result: FilenameAnalysisResult,
                         filesize_result: FilesizeAnalysisResult,
                         message_result: MessageAnalysisResult) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        try:
            summary = {
                'total_files_analyzed': filename_result.total_files,
                'total_messages_analyzed': message_result.total_messages,
                'total_data_size_bytes': filesize_result.total_size_bytes,
                'total_data_size_mb': round(filesize_result.total_size_bytes / (1024 * 1024), 2),
                'duplicate_files_found': filename_result.duplicate_filenames,
                'duplicate_sizes_found': filesize_result.duplicate_sizes,
                'languages_detected': len(message_result.language_distribution),
                'primary_language': max(message_result.language_distribution.items(), key=lambda x: x[1])[0] if message_result.language_distribution else 'unknown',
                'overall_quality_score': self._calculate_overall_quality_score(filename_result, filesize_result, message_result),
                'analysis_completeness': self._calculate_analysis_completeness(filename_result, filesize_result, message_result)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def _generate_key_metrics(self, 
                             filename_result: FilenameAnalysisResult,
                             filesize_result: FilesizeAnalysisResult,
                             message_result: MessageAnalysisResult) -> Dict[str, Any]:
        """Generate key metrics for the analysis."""
        try:
            metrics = {
                'filename_metrics': {
                    'unique_filenames': filename_result.unique_filenames,
                    'duplicate_filenames': filename_result.duplicate_filenames,
                    'quality_score': filename_result.quality_metrics.get('overall_quality', 0.0),
                    'avg_filename_length': filename_result.filename_patterns.get('avg_length', 0),
                    'files_with_special_chars': filename_result.filename_patterns.get('files_with_special_chars', 0)
                },
                'filesize_metrics': {
                    'total_size_mb': round(filesize_result.total_size_bytes / (1024 * 1024), 2),
                    'avg_file_size_mb': round(filesize_result.size_statistics.get('mean_mb', 0), 2),
                    'median_file_size_mb': round(filesize_result.size_statistics.get('median_mb', 0), 2),
                    'largest_file_mb': round(filesize_result.size_statistics.get('max_mb', 0), 2),
                    'size_distribution': filesize_result.size_distribution
                },
                'message_metrics': {
                    'total_messages': message_result.total_messages,
                    'text_messages': message_result.text_messages,
                    'media_messages': message_result.media_messages,
                    'avg_views': round(message_result.engagement_metrics.get('avg_views', 0), 2),
                    'avg_forwards': round(message_result.engagement_metrics.get('avg_forwards', 0), 2),
                    'avg_replies': round(message_result.engagement_metrics.get('avg_replies', 0), 2),
                    'engagement_rate': round(message_result.engagement_metrics.get('engagement_rate', 0), 4)
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to generate key metrics: {e}")
            return {}
    
    def _generate_recommendations(self, 
                                 filename_result: FilenameAnalysisResult,
                                 filesize_result: FilesizeAnalysisResult,
                                 message_result: MessageAnalysisResult) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        try:
            recommendations = []
            
            # Filename recommendations
            if filename_result.recommendations:
                recommendations.extend([f"Filename: {rec}" for rec in filename_result.recommendations])
            
            # Filesize recommendations
            if filesize_result.duplicate_sizes > 0:
                recommendations.append(f"Filesize: {filesize_result.duplicate_sizes} files have duplicate sizes - consider checking for actual duplicates")
            
            if filesize_result.size_statistics.get('max_mb', 0) > 100:
                recommendations.append("Filesize: Some files are very large (>100MB) - consider compression or archiving")
            
            # Message recommendations
            if message_result.quality_metrics.get('text_completeness', 0) < 0.9:
                recommendations.append("Message: Some messages are missing text content - ensure all messages have proper content")
            
            if message_result.engagement_metrics.get('engagement_rate', 0) < 0.01:
                recommendations.append("Message: Low engagement rate detected - consider improving content quality or timing")
            
            # Overall recommendations
            overall_quality = self._calculate_overall_quality_score(filename_result, filesize_result, message_result)
            if overall_quality < 0.7:
                recommendations.append("Overall: Data quality is below optimal - consider implementing data quality improvements")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the analysis report."""
        try:
            metadata = {
                'analysis_version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'config': {
                    'verbose': self.config.verbose,
                    'chunk_size': self.config.chunk_size,
                    'memory_limit': self.config.memory_limit,
                    'items_per_page': self.config.items_per_page,
                    'retry_attempts': self.config.retry_attempts,
                    'retry_delay': self.config.retry_delay
                },
                'system_info': {
                    'python_version': sys.version,
                    'pandas_version': pd.__version__,
                    'platform': sys.platform
                }
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata: {e}")
            return {}
    
    def _calculate_overall_quality_score(self, 
                                        filename_result: FilenameAnalysisResult,
                                        filesize_result: FilesizeAnalysisResult,
                                        message_result: MessageAnalysisResult) -> float:
        """Calculate overall quality score from all analysis results."""
        try:
            filename_quality = filename_result.quality_metrics.get('overall_quality', 0.0)
            message_quality = message_result.quality_metrics.get('overall_quality', 0.0)
            
            # Filesize quality based on distribution and duplicates
            filesize_quality = 1.0
            if filesize_result.duplicate_sizes > 0:
                duplicate_ratio = filesize_result.duplicate_sizes / filesize_result.total_files
                filesize_quality = max(0.0, 1.0 - duplicate_ratio)
            
            # Weighted average
            overall_quality = (
                filename_quality * 0.4 +
                filesize_quality * 0.3 +
                message_quality * 0.3
            )
            
            return round(overall_quality, 3)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall quality score: {e}")
            return 0.0
    
    def _calculate_analysis_completeness(self, 
                                        filename_result: FilenameAnalysisResult,
                                        filesize_result: FilesizeAnalysisResult,
                                        message_result: MessageAnalysisResult) -> float:
        """Calculate analysis completeness score."""
        try:
            total_items = 0
            analyzed_items = 0
            
            # Filename analysis completeness
            if filename_result.total_files > 0:
                total_items += filename_result.total_files
                analyzed_items += filename_result.total_files
            
            # Filesize analysis completeness
            if filesize_result.total_files > 0:
                total_items += filesize_result.total_files
                analyzed_items += filesize_result.total_files
            
            # Message analysis completeness
            if message_result.total_messages > 0:
                total_items += message_result.total_messages
                analyzed_items += message_result.total_messages
            
            if total_items == 0:
                return 0.0
            
            completeness = analyzed_items / total_items
            return round(completeness, 3)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate analysis completeness: {e}")
            return 0.0

# Main entry point for CLI integration
def create_analysis_config(**kwargs) -> AnalysisConfig:
    """Create analysis configuration from kwargs."""
    return AnalysisConfig(**kwargs)

async def run_advanced_intermediate_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Main analysis orchestration function."""
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not config.validate_config():
        raise ValueError("Invalid configuration provided")
    
    logger.info("Starting advanced analysis...")
    logger.info(f"Configuration: {config}")
    
    # Placeholder for full implementation
    return {
        "status": "success",
        "message": "Analysis command foundation implemented",
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat()
    }

# CLI Integration placeholder
def analysis_command(channels: List[str] = None, output_dir: str = None, 
                    enable_file_source: bool = True, enable_api_source: bool = True,
                    enable_diff_analysis: bool = True, verbose: bool = False):
    """CLI command for analysis."""
    try:
        # Create configuration
        config = create_analysis_config(
            channels=channels or [],
            output_dir=output_dir or ANALYSIS_BASE,
            enable_file_source=enable_file_source,
            enable_api_source=enable_api_source,
            enable_diff_analysis=enable_diff_analysis,
            verbose=verbose
        )
        
        # Run analysis - handle both sync and async contexts
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_advanced_intermediate_analysis(config))
                result = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            result = asyncio.run(run_advanced_intermediate_analysis(config))
        
        if verbose:
            print(f"Analysis completed: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis command failed: {e}")
        if verbose:
            print(f"Error: {e}")
        raise

if __name__ == "__main__":
    # Test the basic functionality
    config = create_analysis_config(verbose=True)
    result = asyncio.run(run_advanced_intermediate_analysis(config))
    print(f"Test result: {result}")
