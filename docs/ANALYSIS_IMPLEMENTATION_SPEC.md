# Analysis Command Implementation Specification

## Overview

This document provides detailed implementation specifications for the `analysis` command with specific technology constraints and architectural requirements.

## Implementation Requirements

### Technology Stack
- **pandas**: >=1.5.0 (primary data processing)
- **numpy**: >=1.21.0 (numerical operations)
- **pydantic**: >=2.0.0 (data validation and models)
- **aiohttp**: >=3.8.0 (async HTTP client)
- **langdetect**: >=1.0.9 (language detection)
- **emoji**: >=2.0.0 (emoji analysis)

### Architecture Constraints
1. **Single Module**: All code in `modules/analysis.py` (600-800 lines)
2. **No Caching**: Direct processing without result caching
3. **Pandas-Centric**: All data operations use pandas DataFrames
4. **Pydantic Models**: Data validation and serialization
5. **Pandas JSON Operations**: All JSON file read/write operations must use pandas

## Module Structure

```python
# modules/analysis.py structure
"""
Analysis Command Implementation
Single module containing all analysis functionality
"""

# Imports
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from collections import Counter
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator
from langdetect import detect, LangDetectException
import emoji

# Configuration Constants
# Data Models (Pydantic)
# Data Loading Classes
# Analysis Classes
# Output Classes
# Orchestration Classes
# Main Entry Points
```

## Pydantic Data Models

### Core Models

```python
class AnalysisConfig(BaseModel):
    """Configuration for analysis execution."""
    enable_file_source: bool = True
    enable_api_source: bool = True
    enable_diff_analysis: bool = True
    channels: List[str] = Field(default_factory=list)
    verbose: bool = False
    output_dir: str = "reports/analysis"
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    items_per_page: int = 100
    
    @validator('channels')
    def validate_channels(cls, v):
        if v and not all(ch.startswith('@') for ch in v):
            raise ValueError("Channel names must start with '@'")
        return v

class DataSource(BaseModel):
    """Represents a data source for analysis."""
    source_type: str = Field(..., regex="^(file|api|dual)$")
    channel_name: str
    total_records: int = Field(ge=0)
    date_range: Tuple[Optional[datetime], Optional[datetime]]
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MessageRecord(BaseModel):
    """Normalized message record schema."""
    message_id: Union[int, str]
    channel_username: str
    date: datetime
    text: Optional[str] = None
    creator_username: Optional[str] = None
    creator_first_name: Optional[str] = None
    creator_last_name: Optional[str] = None
    media_type: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = Field(None, ge=0)
    mime_type: Optional[str] = None
    caption: Optional[str] = None
    views: Optional[int] = Field(None, ge=0)
    forwards: Optional[int] = Field(None, ge=0)
    replies: Optional[int] = Field(None, ge=0)
    is_forwarded: Optional[bool] = None
    forwarded_from: Optional[str] = None
    source: str = Field(..., regex="^(file|api)$")
```

### Analysis Result Models

```python
class FilenameAnalysisResult(BaseModel):
    """Filename analysis results."""
    duplicate_filename_detection: Dict[str, Any]
    filename_pattern_analysis: Dict[str, Any]

class FilesizeAnalysisResult(BaseModel):
    """Filesize analysis results."""
    duplicate_filesize_detection: Dict[str, Any]
    filesize_distribution_analysis: Dict[str, Any]

class MessageAnalysisResult(BaseModel):
    """Message analysis results."""
    content_statistics: Dict[str, Any]
    pattern_recognition: Dict[str, Any]
    creator_analysis: Dict[str, Any]
    language_analysis: Dict[str, Any]

class MediaAnalysisResult(BaseModel):
    """Media analysis results."""
    file_size_analysis: Dict[str, Any]
    media_type_analysis: Dict[str, Any]
    filename_analysis: FilenameAnalysisResult
    filesize_analysis: FilesizeAnalysisResult
    media_content_analysis: Dict[str, Any]

class AnalysisResult(BaseModel):
    """Complete analysis result."""
    channel_name: str
    analysis_type: str
    generated_at: datetime
    data_summary: Dict[str, Any]
    analysis_results: Union[MessageAnalysisResult, MediaAnalysisResult, Dict[str, Any]]
    metadata: Dict[str, Any]
```

## Data Loading Implementation

### Base Data Loader

```python
class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def discover_sources(self) -> List[DataSource]:
        """Discover available data sources."""
        raise NotImplementedError
    
    async def load_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from source into DataFrame."""
        raise NotImplementedError
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to common schema using pandas."""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_columns = ['message_id', 'channel_username', 'date', 'source']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Convert date strings to datetime using pandas
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert numeric columns using pandas
        numeric_columns = ['file_size', 'views', 'forwards', 'replies']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        if 'is_forwarded' in df.columns:
            df['is_forwarded'] = df['is_forwarded'].astype(bool)
        
        return df
```

### File Data Loader

```python
class FileDataLoader(BaseDataLoader):
    """Loads data from combined JSON files using pandas."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.collections_dir = Path("reports/collections")
        self.file_pattern = "tg_*_combined.json"
    
    async def discover_sources(self) -> List[DataSource]:
        """Discover available combined JSON files using pandas."""
        sources = []
        
        if not self.collections_dir.exists():
            self.logger.warning(f"Collections directory not found: {self.collections_dir}")
            return sources
        
        for file_path in self.collections_dir.glob(self.file_pattern):
            try:
                # Load JSON data using pandas
                data = pd.read_json(file_path, lines=False)
                
                if data.empty or not isinstance(data, pd.DataFrame):
                    continue
                
                # Extract metadata using pandas
                if 'metadata' in data.columns:
                    metadata_series = data['metadata'].iloc[0] if len(data) > 0 else {}
                    metadata = metadata_series if isinstance(metadata_series, dict) else {}
                else:
                    metadata = {}
                
                channel_name = metadata.get('channel', 'unknown')
                total_messages = metadata.get('total_messages', 0)
                
                if total_messages == 0:
                    continue
                
                # Calculate date range and quality score using pandas
                date_range = self._calculate_date_range_pandas(data)
                quality_score = self._calculate_quality_score_pandas(data)
                
                source = DataSource(
                    source_type="file",
                    channel_name=channel_name,
                    total_records=total_messages,
                    date_range=date_range,
                    quality_score=quality_score,
                    metadata={
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime)
                    }
                )
                sources.append(source)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        return sources
    
    async def load_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from file source using pandas JSON operations."""
        file_path = Path(source.metadata['file_path'])
        
        # Load JSON data using pandas
        data = pd.read_json(file_path, lines=False)
        
        # Extract messages from nested structure using pandas
        if 'messages' in data.columns:
            # Use pandas explode to expand nested messages
            messages_df = data.explode('messages')
            messages_df = messages_df.dropna(subset=['messages'])
            
            # Convert messages to DataFrame using pandas
            messages_list = messages_df['messages'].tolist()
            df = pd.DataFrame(messages_list)
            
            # Add source column
            df['source'] = 'file'
        else:
            # If no messages column, treat the entire DataFrame as messages
            df = data.copy()
            df['source'] = 'file'
        
        return self._normalize_dataframe(df)
    
    def _calculate_date_range_pandas(self, data: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Calculate date range from message data using pandas operations."""
        if data.empty:
            return (None, None)
        
        # Extract dates from messages using pandas
        if 'messages' in data.columns:
            messages_df = data.explode('messages')
            messages_df = messages_df.dropna(subset=['messages'])
            
            # Convert messages to DataFrame and extract dates
            messages_list = messages_df['messages'].tolist()
            messages_df = pd.DataFrame(messages_list)
            
            if 'date' in messages_df.columns:
                dates = pd.to_datetime(messages_df['date'], errors='coerce')
                dates = dates.dropna()
                
                if not dates.empty:
                    return (dates.min(), dates.max())
        
        return (None, None)
    
    def _calculate_quality_score_pandas(self, data: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness using pandas."""
        if data.empty:
            return 0.0
        
        # Extract messages using pandas
        if 'messages' in data.columns:
            messages_df = data.explode('messages')
            messages_df = messages_df.dropna(subset=['messages'])
            
            # Convert messages to DataFrame
            messages_list = messages_df['messages'].tolist()
            messages_df = pd.DataFrame(messages_list)
            
            if messages_df.empty:
                return 0.0
            
            # Check for required fields using pandas
            required_fields = ['message_id', 'channel_username', 'date']
            complete_messages = 0
            
            for field in required_fields:
                if field in messages_df.columns:
                    complete_messages += messages_df[field].notna().sum()
            
            total_messages = len(messages_df)
            return complete_messages / (total_messages * len(required_fields)) if total_messages > 0 else 0.0
        
        return 0.0
```

## Analysis Implementation

### Filename Analyzer

```python
class FilenameAnalyzer:
    """Analyzes filename patterns and duplicates using pandas."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> FilenameAnalysisResult:
        """Perform filename analysis using pandas operations."""
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
        """Analyze duplicate filenames using pandas value_counts."""
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
        """Analyze filename patterns using pandas string operations."""
        filenames = df['file_name']
        
        # Filename length analysis using pandas
        filename_lengths = filenames.str.len()
        length_stats = {
            "count": len(filename_lengths),
            "min": int(filename_lengths.min()),
            "max": int(filename_lengths.max()),
            "mean": round(filename_lengths.mean(), 2),
            "median": int(filename_lengths.median())
        }
        
        # Extension analysis using pandas
        extensions = filenames.str.extract(r'\.([^.]+)$')[0].str.lower()
        extension_counts = extensions.value_counts()
        common_extensions = [
            {"ext": f".{ext}", "count": count}
            for ext, count in extension_counts.head(10).items()
        ]
        
        # Special character analysis using pandas
        special_char_pattern = r'[^\w\s.-]'
        files_with_special_chars = filenames.str.contains(special_char_pattern, regex=True).sum()
        
        # Space analysis using pandas
        files_with_spaces = filenames.str.contains(' ').sum()
        
        return {
            "filename_length": length_stats,
            "common_extensions": common_extensions,
            "files_with_special_chars": int(files_with_special_chars),
            "files_with_spaces": int(files_with_spaces)
        }
```

### Filesize Analyzer

```python
class FilesizeAnalyzer:
    """Analyzes filesize patterns and duplicates using pandas."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, df: pd.DataFrame, source: DataSource) -> FilesizeAnalysisResult:
        """Perform filesize analysis using pandas operations."""
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
        """Analyze duplicate filesizes using pandas value_counts."""
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
        """Analyze filesize distribution using pandas cut."""
        file_sizes = df['file_size']
        
        # Size distribution bins using pandas cut
        size_bins = pd.cut(
            file_sizes,
            bins=[0, 1024*1024, 5*1024*1024, 10*1024*1024, float('inf')],
            labels=['0-1MB', '1-5MB', '5-10MB', '10MB+']
        )
        size_distribution = size_bins.value_counts().to_dict()
        
        # Potential duplicates by size using pandas groupby
        size_groups = df.groupby('file_size')
        potential_duplicates = []
        
        for size, group in size_groups:
            if len(group) > 1:  # Multiple files with same size
                filenames = group['file_name'].tolist()
                potential_duplicates.append({
                    "size_bytes": int(size),
                    "files": filenames
                })
        
        # Sort by size and limit to top 10
        potential_duplicates.sort(key=lambda x: x['size_bytes'], reverse=True)
        potential_duplicates = potential_duplicates[:10]
        
        return {
            "size_frequency_distribution": size_distribution,
            "potential_duplicates_by_size": potential_duplicates
        }
```

## Main Entry Points

```python
def create_analysis_config(**kwargs) -> AnalysisConfig:
    """Create analysis configuration from kwargs."""
    return AnalysisConfig(**kwargs)

async def run_advanced_intermediate_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Main analysis orchestration function."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize data loaders
        file_loader = FileDataLoader(config)
        api_loader = ApiDataLoader(config)
        
        # Discover sources
        sources = []
        if config.enable_file_source:
            file_sources = await file_loader.discover_sources()
            sources.extend(file_sources)
        
        if config.enable_api_source:
            api_sources = await api_loader.discover_sources()
            sources.extend(api_sources)
        
        if not sources:
            return {"error": "No data sources available"}
        
        # Filter sources by channels if specified
        if config.channels:
            sources = [s for s in sources if s.channel_name in config.channels]
        
        results = {}
        
        # Process each source
        for source in sources:
            logger.info(f"Processing source: {source.channel_name}")
            
            # Load data
            if source.source_type == "file":
                df = await file_loader.load_data(source)
            elif source.source_type == "api":
                df = await api_loader.load_data(source)
            else:
                continue
            
            # Run analysis
            filename_analyzer = FilenameAnalyzer(config)
            filesize_analyzer = FilesizeAnalyzer(config)
            
            filename_result = filename_analyzer.analyze(df, source)
            filesize_result = filesize_analyzer.analyze(df, source)
            
            # Store results
            results[source.channel_name] = {
                "filename_analysis": filename_result.dict(),
                "filesize_analysis": filesize_result.dict(),
                "metadata": {
                    "source_type": source.source_type,
                    "total_records": len(df),
                    "processed_at": datetime.now().isoformat()
                }
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}
```

## JSON Output Operations with Pandas

### Output Generation

```python
class JsonOutputManager:
    """Manages JSON output operations using pandas."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analysis_results(self, results: Dict[str, Any], filename: str) -> str:
        """Save analysis results to JSON file using pandas."""
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert results to DataFrame for processing
        results_df = pd.DataFrame([results])
        
        # Use pandas to_json for output
        results_df.to_json(
            output_path,
            orient='records',
            indent=2,
            date_format='iso',
            force_ascii=False
        )
        
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
        
        # Convert current results to DataFrame
        current_df = pd.DataFrame([current_results])
        
        # Perform comparison using pandas operations
        comparison = {
            "current_count": len(current_df),
            "previous_count": len(previous_results),
            "difference": len(current_df) - len(previous_results),
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison
```

### Configuration File Operations

```python
def save_config_to_json(config: AnalysisConfig, filename: str = "analysis_config.json") -> str:
    """Save configuration to JSON file using pandas."""
    config_dict = config.dict()
    config_df = pd.DataFrame([config_dict])
    
    output_path = Path("reports/analysis") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_df.to_json(
        output_path,
        orient='records',
        indent=2,
        force_ascii=False
    )
    
    return str(output_path)

def load_config_from_json(filename: str = "analysis_config.json") -> AnalysisConfig:
    """Load configuration from JSON file using pandas."""
    config_path = Path("reports/analysis") / filename
    
    if not config_path.exists():
        return AnalysisConfig()
    
    config_df = pd.read_json(config_path, orient='records')
    config_dict = config_df.iloc[0].to_dict()
    
    return AnalysisConfig(**config_dict)
```

## Performance Considerations

### Memory Management
- Use pandas chunking for large datasets
- Process data in batches to avoid memory exhaustion
- Use appropriate pandas data types (category, int32, etc.)
- Use pandas `read_json` with `chunksize` parameter for large JSON files

### Vectorized Operations
- Prefer pandas vectorized operations over Python loops
- Use pandas string methods for text processing
- Leverage pandas groupby and aggregation functions
- Use pandas `to_json` and `read_json` for all JSON operations

### JSON Operations with Pandas
- Use `pd.read_json()` for all JSON file reading operations
- Use `pd.to_json()` for all JSON file writing operations
- Leverage pandas `orient` parameter for different JSON formats
- Use pandas `date_format` parameter for consistent datetime handling
- Apply pandas `force_ascii=False` for proper Unicode support

### Error Handling
- Validate data using pydantic models
- Handle missing data gracefully with pandas
- Provide meaningful error messages and logging
- Handle JSON parsing errors with pandas error handling

## Testing Strategy

### Unit Tests
- Test individual analyzers with mock data
- Validate pydantic models with various inputs
- Test pandas operations with edge cases

### Integration Tests
- Test full pipeline with sample data
- Validate output format and structure
- Test error handling and recovery

### Performance Tests
- Test with large datasets (100k+ records)
- Measure memory usage and processing time
- Validate performance requirements

## Conclusion

This implementation specification provides a complete guide for building the analysis command using pandas, numpy, and pydantic in a single module architecture without caching. The design emphasizes performance, maintainability, and comprehensive analysis capabilities.
