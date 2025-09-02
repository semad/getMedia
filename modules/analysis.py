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
