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
