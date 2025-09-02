# Analysis Command Design Specification

## Overview

The `analysis` command is a redesigned, modern analysis tool that generates comprehensive insights from Telegram message data. It replaces the previous `report` command with improved architecture, better performance, and enhanced user experience.

## Design Goals

### Primary Objectives
1. **Unified Analysis Interface** - Single command for all analysis types
2. **Performance Optimization** - Faster processing with better resource management
3. **Modular Architecture** - Clean separation of concerns and easy maintenance
4. **User Experience** - Intuitive CLI with helpful feedback and progress tracking
5. **Extensibility** - Easy to add new analysis types and fields
6. **Output Consistency** - Single JSON format for all analysis results

### Success Metrics
- **Processing Speed**: 50% faster than previous report command
- **Memory Usage**: 30% reduction in peak memory consumption
- **Code Maintainability**: Reduced complexity and improved testability
- **User Satisfaction**: Clear feedback and error handling

## Command Architecture

### Core Components

```
analysis/
├── __init__.py                 # Package initialization
├── core/
│   ├── __init__.py            # Core module exports
│   ├── analyzer.py            # Main analysis orchestrator
│   ├── data_loader.py         # Data loading abstraction
│   ├── processor.py           # Data processing pipeline
│   └── output_manager.py      # Output generation and management
├── processors/
│   ├── __init__.py            # Processor module exports
│   ├── file_processor.py      # File-based data processing
│   └── db_processor.py        # Database API processing
├── analyzers/
│   ├── __init__.py            # Analyzer module exports
│   ├── message_analyzer.py    # Message content analysis
│   ├── media_analyzer.py      # Media file analysis and size optimization
│   ├── temporal_analyzer.py   # Time-based analysis
│   ├── engagement_analyzer.py # User engagement metrics
│   └── network_analyzer.py    # Network and relationship analysis
├── outputs/
│   ├── __init__.py            # Output module exports
│   └── json_formatter.py      # JSON output formatting using pandas
└── utils/
    ├── __init__.py            # Utility module exports
    ├── progress.py            # Progress tracking and display
    ├── validation.py          # Input validation and sanitization
    └── caching.py             # Data caching and optimization
```

## Command Interface

### Basic Syntax
```bash
python main.py analysis [OPTIONS] [ANALYSIS_TYPE]
```

### Command Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--source` | `-s` | string | `auto` | Data source: `file`, `db`, `auto` |
| `--channels` | `-c` | string | `all` | Comma-separated channel list or `all` |
| `--time-range` | `-t` | string | `all` | Time range: `all`, `last_week`, `last_month`, `custom` |
| `--custom-start` | | string | | Custom start date (YYYY-MM-DD) |
| `--custom-end` | | string | | Custom end date (YYYY-MM-DD) |
| `--filters` | `-f` | string | | JSON string of custom filters |
| `--parallel` | `-p` | integer | `1` | Number of parallel processing threads |
| `--cache` | | flag | `False` | Enable result caching |
| `--verbose` | `-v` | flag | `False` | Enable verbose logging |
| `--help` | `-h` | flag | | Show help message |

### Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| `messages` | Message content and metadata analysis | Content insights, text analysis |
| `media` | Media file analysis and statistics | File types, sizes, formats, storage analysis |
| `temporal` | Time-based patterns and trends | Activity patterns, seasonal analysis |
| `engagement` | User interaction and engagement metrics | Popularity, reach analysis |
| `network` | Channel relationships and network analysis | Cross-channel insights |
| `comprehensive` | All analysis types combined | Complete channel overview |

### Media Analysis Details

The `media` analysis type provides comprehensive insights into media files:

#### **File Size Analysis**
- **Total Storage**: Aggregate file sizes across all media
- **Size Distribution**: Histogram of file sizes (small, medium, large)
- **Average Sizes**: Mean, median, and mode file sizes by type
- **Size Trends**: File size patterns over time
- **Storage Efficiency**: Compression ratios and optimization opportunities

#### **Media Type Analysis**
- **File Formats**: Distribution of image, video, audio, document types
- **Format Popularity**: Most common media formats used
- **Quality Metrics**: Resolution, bitrate, duration analysis
- **Compatibility**: Cross-platform format support assessment

#### **Storage Impact Analysis**
- **Channel Storage**: Per-channel media storage requirements
- **Growth Patterns**: Media storage growth over time
- **Cost Implications**: Storage cost projections and optimization
- **Cleanup Opportunities**: Identify unused or duplicate media

## Data Flow Architecture

### 1. Command Parsing and Validation
```
CLI Input → Option Validation → Configuration Building → Source Detection
```

### 2. Data Loading Layer
```
Source Selection → Data Loader → Connection Management → Data Retrieval → pd.DataFrame
```

### 3. Processing Pipeline
```
pd.DataFrame → Data Cleaning → Preprocessing → Analysis Execution → Result Aggregation
```

### 4. Output Generation
```
Analysis Results → JSON Formatting → File Management → Output Files
```

### Data Type Consistency
- **Input**: JSON files or Database API responses
- **Processing**: pandas DataFrame throughout the pipeline
- **Output**: Structured JSON files using pandas DataFrame.to_json()
- **Memory**: Chunked processing for large datasets

## Implementation Details

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

# Configure logging
logger = logging.getLogger(__name__)
```

### Core Analyzer Class

```python
class AnalysisOrchestrator:
    """Main orchestrator for analysis operations."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.data_loader = self._create_data_loader()
        self.processors = self._create_processors()
        self.output_manager = self._create_output_manager()
    
    async def run_analysis(self, analysis_type: str) -> AnalysisResult:
        """Execute the complete analysis pipeline."""
        # 1. Load data
        data = await self.data_loader.load_data()
        
        # 2. Process data
        processed_data = await self._process_data(data)
        
        # 3. Run analysis
        results = await self._run_analysis(processed_data, analysis_type)
        
        # 4. Generate output
        output_files = await self.output_manager.generate_output(results)
        
        return AnalysisResult(
            success=True,
            data=results,
            output_files=output_files,
            metadata=self._generate_metadata()
        )
    
    def _create_data_loader(self) -> DataLoader:
        """Create appropriate data loader based on configuration."""
        if self.config.source_type == "file":
            return FileDataLoader(self.config)
        elif self.config.source_type == "db":
            return DatabaseDataLoader(self.config)
        else:
            # Auto-detect: try file first, then database
            return FileDataLoader(self.config)
    
    def _create_processors(self) -> Dict[str, Any]:
        """Create analysis processors."""
        return {
            "messages": MessageAnalyzer(self.config),
            "media": MediaAnalyzer(self.config),
            "temporal": TemporalAnalyzer(self.config),
            "engagement": EngagementAnalyzer(self.config),
            "network": NetworkAnalyzer(self.config)
        }
    
    def _create_output_manager(self) -> OutputManager:
        """Create output manager."""
        return OutputManager(self.config)
    
    async def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the data."""
        if data.empty:
            return data
        
        # Basic data cleaning
        processed_data = data.copy()
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates()
        
        # Handle missing values
        processed_data = processed_data.fillna("")
        
        # Convert date columns if they exist
        date_columns = [col for col in processed_data.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
            except:
                pass
        
        return processed_data
    
    async def _run_analysis(self, data: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Run the specified analysis type."""
        if analysis_type == "comprehensive":
            return await self._run_comprehensive_analysis(data)
        elif analysis_type in self.processors:
            processor = self.processors[analysis_type]
            return await processor.analyze(data)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    async def _run_comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all analysis types and combine results."""
        results = {}
        
        for analysis_type, processor in self.processors.items():
            try:
                result = await processor.analyze(data)
                results[analysis_type] = result
            except Exception as e:
                logger.warning(f"Failed to run {analysis_type} analysis: {e}")
                results[analysis_type] = {"error": str(e)}
        
        return results
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the analysis."""
        return {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "source_type": self.config.source_type,
                "parallel_workers": self.config.parallel_workers,
                "batch_size": self.config.batch_size,
                "enable_caching": self.config.enable_caching
            }
        }
```

### Data Loading Abstraction

```python
# Type aliases for clarity
ProcessedData = pd.DataFrame
RawData = pd.DataFrame

class DataLoader:
    """Abstract base class for data loading."""
    
    @abstractmethod
    async def load_data(self) -> pd.DataFrame:
        """Load raw data from source and return as pandas DataFrame."""
        pass
    
    @abstractmethod
    async def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate loaded DataFrame data."""
        pass

class FileDataLoader(DataLoader):
    """Load data from JSON files using pandas for optimal performance."""
    
    async def load_data(self) -> pd.DataFrame:
        """Load JSON files using pandas read_json for memory efficiency."""
        json_files = self._discover_json_files()
        
        # Use pandas for efficient JSON reading with proper chunking for large files
        dataframes = []
        for file_path in json_files:
            try:
                # Check file size to determine reading strategy
                file_size = Path(file_path).stat().st_size
                
                if file_size > 100 * 1024 * 1024:  # > 100MB
                    # Large file: use manual chunking with jsonlines
                    df = await self._read_large_json_file(file_path)
                else:
                    # Small file: use pandas read_json directly
                    df = pd.read_json(file_path, lines=True)
                
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
        
        # Combine all dataframes efficiently
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        return pd.DataFrame()
    
    async def _read_large_json_file(self, file_path: Path) -> pd.DataFrame:
        """Read large JSON files using chunked processing."""
        
        chunk_size = 10000
        dataframes = []
        
        try:
            with open(file_path, 'r') as f:
                chunk_data = []
                for line_num, line in enumerate(f):
                    try:
                        chunk_data.append(json.loads(line.strip()))
                        
                        if len(chunk_data) >= chunk_size:
                            # Process chunk
                            chunk_df = pd.DataFrame(chunk_data)
                            dataframes.append(chunk_df)
                            chunk_data = []
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
                
                # Process remaining data
                if chunk_data:
                    chunk_df = pd.DataFrame(chunk_data)
                    dataframes.append(chunk_df)
            
            # Combine all chunks
            if dataframes:
                return pd.concat(dataframes, ignore_index=True)
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to read large file {file_path}: {e}")
            return pd.DataFrame()

class DatabaseDataLoader(DataLoader):
    """Load data from database API and convert to pandas DataFrame."""
    
    async def load_data(self) -> pd.DataFrame:
        """Fetch data from API and convert to pandas DataFrame."""
        # Fetch data from API endpoints
        raw_data = await self._fetch_api_data()
        
        # Convert to pandas DataFrame for efficient processing
        return pd.DataFrame(raw_data)
```

### Progress Tracking Implementation

```python
class ProgressTracker:
    """Tracks and displays progress for long-running operations."""
    
    def __init__(self):
        self.current_operation = ""
        self.progress_bar = None
    
    def track(self, operation_name: str):
        """Context manager for tracking operations."""
        return ProgressContext(self, operation_name)
    
    def update_progress(self, current: int, total: int, description: str = ""):
        """Update progress display."""
        if self.progress_bar:
            self.progress_bar.update(current - self.progress_bar.n)
            self.progress_bar.set_description(description)

class ProgressContext:
    """Context manager for progress tracking."""
    
    def __init__(self, tracker: ProgressTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
    
    def __enter__(self):
        self.tracker.current_operation = self.operation_name
        print(f"🔄 {self.operation_name}")
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print(f"✅ {self.operation_name} completed")
        else:
            print(f"❌ {self.operation_name} failed: {exc_val}")

### Analysis Pipeline

```python
class AnalysisPipeline:
    """Executes analysis pipeline with caching and optimization."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache = CacheManager()
        self.progress = ProgressTracker()
    
    async def execute(self, data: ProcessedData, analysis_type: str) -> AnalysisResult:
        """Execute analysis with progress tracking and caching."""
        
        # Check cache first
        cache_key = self._generate_cache_key(data, analysis_type)
        if cached_result := self.cache.get(cache_key):
            return cached_result
        
        # Execute analysis
        with self.progress.track(f"Running {analysis_type} analysis"):
            result = await self._run_analysis(data, analysis_type)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, data: ProcessedData, analysis_type: str) -> str:
        """Generate a cache key for the analysis."""
        # Simple hash-based cache key
        data_hash = hash(str(data.shape) + str(data.columns.tolist()))
        return f"{analysis_type}_{data_hash}"

### Cache Manager Implementation

```python
class CacheManager:
    """Manages caching of analysis results."""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        self.timestamps = {}
    
    def get(self, key: str):
        """Get cached result if valid."""
        if key in self.cache:
            if self._is_valid(key):
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache a result with timestamp."""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def _is_valid(self, key: str) -> bool:
        """Check if cached item is still valid."""
        if key not in self.timestamps:
            return False
        return (time.time() - self.timestamps[key]) < self.ttl
    
    def clear_expired(self):
        """Clear expired cache entries."""
        expired_keys = [
            key for key in self.timestamps 
            if not self._is_valid(key)
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
```

### Output Manager Implementation

```python
class OutputManager:
    """Manages output generation and file management."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.json_formatter = PandasJsonFormatter(config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_output(self, analysis_result: AnalysisResult) -> List[Path]:
        """Generate all output files for the analysis results."""
        
        output_files = []
        
        # Generate output for each channel
        for channel, channel_data in analysis_result.data.items():
            channel_files = await self.json_formatter.format_and_save(
                analysis_result, channel
            )
            output_files.extend(channel_files)
        
        # Generate consolidated summary
        summary_file = await self._generate_consolidated_summary(analysis_result)
        output_files.append(summary_file)
        
        return output_files
    
    async def _generate_consolidated_summary(self, analysis_result: AnalysisResult) -> Path:
        """Generate a consolidated summary across all channels."""
        
        summary_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_type": analysis_result.analysis_type,
                "source": analysis_result.source,
                "total_channels": len(analysis_result.data),
                "processing_time": analysis_result.processing_time
            },
            "overall_summary": {
                "total_messages": analysis_result.message_count,
                "total_media": analysis_result.media_count,
                "total_size_bytes": analysis_result.total_size,
                "channels_analyzed": list(analysis_result.data.keys())
            },
            "per_channel_summary": analysis_result.data
        }
        
        summary_path = self.output_dir / "consolidated_summary.json"
        
        try:
            # Use pandas for JSON output
            df = pd.DataFrame([summary_data])
            df.to_json(
                summary_path,
                orient='records',
                indent=2,
                date_format='iso',
                default_handler=str,
                index=False
            )
            
            logger.info(f"✅ Generated consolidated summary: {summary_path}")
            return summary_path
            
        except Exception as e:
            logger.error(f"❌ Failed to generate consolidated summary: {e}")
            raise
```

### JSON Output Formatter with Pandas

```python
class PandasJsonFormatter:
    """JSON output formatter using pandas for optimal performance and memory efficiency."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def format_and_save(self, analysis_result: AnalysisResult, channel: str) -> List[Path]:
        """Format analysis results and save as JSON using pandas."""
        
        output_files = []
        
        # 1. Main Analysis Results (JSON)
        main_results = self._prepare_main_results(analysis_result, channel)
        main_file = await self._save_json_with_pandas(main_results, f"{channel}_analysis.json")
        output_files.append(main_file)
        
        # 2. Summary Statistics (JSON)
        summary_data = self._prepare_summary_data(analysis_result, channel)
        summary_file = await self._save_json_with_pandas(summary_data, f"{channel}_summary.json")
        output_files.append(summary_file)
        
        # 3. Raw Data Export (if requested)
        if self.config.include_raw_data:
            raw_data = self._prepare_raw_data(analysis_result, channel)
            raw_file = await self._save_json_with_pandas(raw_data, f"{channel}_raw_data.json")
            output_files.append(raw_file)
        
        return output_files
    
    async def _save_json_with_pandas(self, data: Dict, filename: str) -> Path:
        """Save data as JSON using pandas for optimal performance."""
        
        output_path = self.output_dir / filename
        
        try:
            # Convert to pandas DataFrame for efficient JSON writing
            if isinstance(data, dict):
                # For nested dictionaries, flatten or structure appropriately
                df = self._dict_to_dataframe(data)
            else:
                df = pd.DataFrame(data)
            
            # Use pandas to_json with optimized settings
            df.to_json(
                output_path,
                orient='records',
                indent=2,
                date_format='iso',
                default_handler=str,  # Handle non-serializable objects
                compression=None,     # No compression for better compatibility
                index=False          # Don't include DataFrame index
            )
            
            logger.info(f"✅ Saved JSON output: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save JSON: {e}")
            raise
    
    def _dict_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """Convert nested dictionary to pandas DataFrame efficiently."""
        
        # Handle different data structures
        if isinstance(data, dict):
            # For simple key-value pairs
            if all(isinstance(v, (str, int, float, bool)) for v in data.values()):
                return pd.DataFrame([data])
            
            # For nested structures, flatten or create appropriate structure
            flattened = self._flatten_dict(data)
            return pd.DataFrame([flattened])
        
        return pd.DataFrame(data)
    
    def _flatten_dict(self, data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for DataFrame conversion."""
        
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Handle lists by creating separate entries or flattening
                if v and isinstance(v[0], dict):
                    # List of dictionaries - create separate rows
                    for i, item in enumerate(v):
                        item['_list_index'] = i
                        items.extend(self._flatten_dict(item, new_key, sep=sep).items())
                else:
                    # Simple list - join as string
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _prepare_main_results(self, analysis_result: AnalysisResult, channel: str) -> Dict:
        """Prepare main analysis results for JSON output."""
        
        return {
            "metadata": {
                "channel": channel,
                "generated_at": datetime.now().isoformat(),
                "analysis_type": analysis_result.analysis_type,
                "source": analysis_result.source,
                "processing_time": analysis_result.processing_time
            },
            "results": analysis_result.data,
            "summary": analysis_result.summary
        }
    
    def _prepare_summary_data(self, analysis_result: AnalysisResult, channel: str) -> Dict:
        """Prepare summary statistics for JSON output."""
        
        return {
            "channel": channel,
            "generated_at": datetime.now().isoformat(),
            "summary": analysis_result.summary,
            "file_info": {
                "total_size_bytes": analysis_result.total_size,
                "message_count": analysis_result.message_count,
                "media_count": analysis_result.media_count
            }
        }
    
    def _prepare_raw_data(self, analysis_result: AnalysisResult, channel: str) -> Dict:
        """Prepare raw data for JSON output."""
        
        return {
            "channel": channel,
            "generated_at": datetime.now().isoformat(),
            "raw_data": {
                "total_records": analysis_result.message_count,
                "data_sample": analysis_result.data.get(channel, {}),
                "metadata": analysis_result.metadata
            }
        }
```

### Base Analyzer Class

```python
class BaseAnalyzer:
    """Base class for all analyzers."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the data and return results."""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data meets minimum requirements."""
        if data.empty:
            return False
        return True

class MessageAnalyzer(BaseAnalyzer):
    """Analyzes message content and metadata."""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze message data."""
        if not self._validate_data(data):
            return {"error": "No valid data to analyze"}
        
        return {
            "total_messages": len(data),
            "message_types": self._analyze_message_types(data),
            "content_analysis": self._analyze_content(data),
            "metadata_analysis": self._analyze_metadata(data)
        }
    
    def _analyze_message_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different types of messages."""
        # Implementation details
        return {"text": 0, "media": 0, "other": 0}
    
    def _analyze_content(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze message content."""
        # Implementation details
        return {"avg_length": 0, "language_distribution": {}}
    
    def _analyze_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze message metadata."""
        # Implementation details
        return {"date_range": "", "user_distribution": {}}

class TemporalAnalyzer(BaseAnalyzer):
    """Analyzes time-based patterns and trends."""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns."""
        if not self._validate_data(data):
            return {"error": "No valid data to analyze"}
        
        return {
            "time_distribution": self._analyze_time_distribution(data),
            "seasonal_patterns": self._analyze_seasonal_patterns(data),
            "activity_trends": self._analyze_activity_trends(data)
        }
    
    def _analyze_time_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of messages over time."""
        # Implementation details
        return {"hourly": {}, "daily": {}, "monthly": {}}
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in data."""
        # Implementation details
        return {"seasonal_trends": {}, "monthly_patterns": {}}
    
    def _analyze_activity_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze activity trends over time."""
        # Implementation details
        return {"trends": {}, "growth_rate": 0.0}

class EngagementAnalyzer(BaseAnalyzer):
    """Analyzes user engagement metrics."""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement patterns."""
        if not self._validate_data(data):
            return {"error": "No valid data to analyze"}
        
        return {
            "engagement_metrics": self._analyze_engagement_metrics(data),
            "user_activity": self._analyze_user_activity(data),
            "interaction_patterns": self._analyze_interaction_patterns(data)
        }
    
    def _analyze_engagement_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        # Implementation details
        return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
    
    def _analyze_user_activity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        # Implementation details
        return {"active_users": 0, "user_engagement_rate": 0.0}
    
    def _analyze_interaction_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze interaction patterns."""
        # Implementation details
        return {"interaction_types": {}, "response_times": {}}

class NetworkAnalyzer(BaseAnalyzer):
    """Analyzes channel relationships and network patterns."""
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze network patterns."""
        if not self._validate_data(data):
            return {"error": "No valid data to analyze"}
        
        return {
            "channel_relationships": self._analyze_channel_relationships(data),
            "cross_channel_patterns": self._analyze_cross_channel_patterns(data)
        }
    
    def _analyze_channel_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between channels."""
        # Implementation details
        return {"related_channels": [], "relationship_strength": {}}
    
    def _analyze_cross_channel_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns across multiple channels."""
        # Implementation details
        return {"cross_channel_trends": {}, "shared_patterns": {}}

### Media Analyzer Implementation

```python
class MediaAnalyzer(BaseAnalyzer):
    """Comprehensive media file analysis with size optimization insights."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.size_categories = {
            'small': (0, 1024 * 1024),      # 0-1MB
            'medium': (1024 * 1024, 10 * 1024 * 1024),  # 1-10MB
            'large': (10 * 1024 * 1024, 100 * 1024 * 1024),  # 10-100MB
            'huge': (100 * 1024 * 1024, float('inf'))   # 100MB+
        }
    
    async def analyze_media(self, data: ProcessedData) -> MediaAnalysisResult:
        """Perform comprehensive media analysis."""
        
        media_data = self._extract_media_messages(data)
        
        return MediaAnalysisResult(
            file_size_analysis=self._analyze_file_sizes(media_data),
            media_type_analysis=self._analyze_media_types(media_data),
            storage_impact=self._analyze_storage_impact(media_data),
            optimization_recommendations=self._generate_optimization_recommendations(media_data)
        )
    
    def _analyze_file_sizes(self, media_data: List[MediaMessage]) -> FileSizeAnalysis:
        """Analyze file sizes and distribution patterns."""
        
        sizes = [msg.file_size for msg in media_data if msg.file_size]
        
        return FileSizeAnalysis(
            total_size=sum(sizes),
            count=len(sizes),
            average_size=sum(sizes) / len(sizes) if sizes else 0,
            median_size=self._calculate_median(sizes),
            size_distribution=self._categorize_sizes(sizes),
            size_trends=self._analyze_size_trends(media_data),
            storage_efficiency=self._calculate_storage_efficiency(media_data)
        )
    
    def _categorize_sizes(self, sizes: List[int]) -> Dict[str, Dict]:
        """Categorize files by size ranges."""
        
        categories = {}
        for category, (min_size, max_size) in self.size_categories.items():
            category_sizes = [s for s in sizes if min_size <= s < max_size]
            categories[category] = {
                'count': len(category_sizes),
                'total_size': sum(category_sizes),
                'average_size': sum(category_sizes) / len(category_sizes) if category_sizes else 0,
                'percentage': (len(category_sizes) / len(sizes)) * 100 if sizes else 0
            }
        
        return categories
    
    def _generate_optimization_recommendations(self, media_data: List[MediaMessage]) -> List[str]:
        """Generate storage optimization recommendations."""
        
        recommendations = []
        
        # Analyze large files
        large_files = [msg for msg in media_data if msg.file_size > 10 * 1024 * 1024]
        if large_files:
            recommendations.append(f"Found {len(large_files)} large files (>10MB) - consider compression")
        
        # Analyze duplicate patterns
        duplicates = self._find_duplicate_patterns(media_data)
        if duplicates:
            recommendations.append(f"Potential {len(duplicates)} duplicate files - review for cleanup")
        
        # Storage growth analysis
        growth_rate = self._calculate_storage_growth_rate(media_data)
        if growth_rate > 0.1:  # 10% monthly growth
            recommendations.append(f"High storage growth rate ({growth_rate:.1%}) - implement retention policies")
        
        return recommendations
    
    def _find_duplicate_patterns(self, media_data: List[MediaMessage]) -> List[Dict]:
        """Find potential duplicate files based on various criteria."""
        # Implementation details
        return []
    
    def _calculate_storage_growth_rate(self, media_data: List[MediaMessage]) -> float:
        """Calculate monthly storage growth rate."""
        # Implementation details
        return 0.0
    
    def _analyze_media_types(self, media_data: List[MediaMessage]) -> Dict[str, Any]:
        """Analyze media types and formats."""
        # Implementation details
        return {"formats": {}, "quality_metrics": {}}
    
    def _analyze_storage_impact(self, media_data: List[MediaMessage]) -> Dict[str, Any]:
        """Analyze storage impact and growth patterns."""
        # Implementation details
        return {"storage_trends": {}, "cost_implications": {}}
    
    def _calculate_median(self, sizes: List[int]) -> float:
        """Calculate median file size."""
        if not sizes:
            return 0.0
        sorted_sizes = sorted(sizes)
        mid = len(sorted_sizes) // 2
        if len(sorted_sizes) % 2 == 0:
            return (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2
        return sorted_sizes[mid]
    
    def _analyze_size_trends(self, media_data: List[MediaMessage]) -> Dict[str, Any]:
        """Analyze file size trends over time."""
        # Implementation details
        return {"trends": {}, "growth_patterns": {}}
    
    def _calculate_storage_efficiency(self, media_data: List[MediaMessage]) -> float:
        """Calculate storage efficiency score."""
        # Implementation details
        return 0.85  # Example efficiency score
```

## Configuration Management

### Analysis Configuration

```python
@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    
    # Data source configuration
    source_type: str = "auto"
    source_path: Optional[str] = None
    db_url: Optional[str] = None
    
    # Processing configuration
    parallel_workers: int = 1
    batch_size: int = 1000
    enable_caching: bool = False
    cache_ttl: int = 3600
    
    # Output configuration
    output_dir: str = "reports/analysis"
    include_summary: bool = True
    include_raw_data: bool = False
    
    # Filtering configuration
    time_range: Optional[str] = None
    custom_filters: Optional[Dict] = None
    channel_whitelist: Optional[List[str]] = None
    channel_blacklist: Optional[List[str]] = None

# Core Data Structures
@dataclass
class ValidationResult:
    """Data validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    data_quality_score: float
    validation_timestamp: str

@dataclass
class AnalysisResult:
    """Complete analysis results."""
    success: bool
    data: Dict[str, Any]
    output_files: List[Path]
    metadata: Dict[str, Any]
    analysis_type: str
    source: str
    processing_time: float
    total_size: int
    message_count: int
    media_count: int
    summary: Dict[str, Any]

# Data Type Definitions
@dataclass
class MediaMessage:
    """Represents a media message with file information."""
    message_id: str
    file_size: int
    file_type: str
    timestamp: str
    channel: str
    user_id: str
    media_type: str  # image, video, audio, document

# Media Analysis Data Structures
@dataclass
class FileSizeAnalysis:
    """File size analysis results."""
    total_size: int
    count: int
    average_size: float
    median_size: float
    size_distribution: Dict[str, Dict]
    size_trends: Dict[str, Any]
    storage_efficiency: float

@dataclass
class MediaAnalysisResult:
    """Complete media analysis results."""
    file_size_analysis: FileSizeAnalysis
    media_type_analysis: Dict[str, Any]
    storage_impact: Dict[str, Any]
    optimization_recommendations: List[str]
```

### Environment Configuration

```python
# config.py additions
ANALYSIS_CONFIG = {
    "DEFAULT_OUTPUT_DIR": "reports/analysis",
    "DEFAULT_CACHE_TTL": 3600,
    "MAX_PARALLEL_WORKERS": 4,
    "DEFAULT_BATCH_SIZE": 1000,
    "ENABLE_PROGRESS_BARS": True,
    "LOG_LEVEL": "INFO",
    "PANDAS_CONFIG": {
        "CHUNK_SIZE": 10000,           # Manual chunking size for large files
        "MAX_MEMORY_USAGE": "2GB",     # Memory limit for large files
        "ENABLE_CHUNKED_READING": True, # Use manual chunking for large files
        "JSON_ORIENT": "records",      # JSON output orientation
        "COMPRESSION": None,           # No compression for compatibility
        "DATE_FORMAT": "iso"           # ISO format for dates
    }
}
```

### Pandas Dependencies and Requirements

```python
# requirements.txt additions
pandas>=2.0.0          # Core data processing
numpy>=1.24.0          # Numerical operations
pyarrow>=12.0.0        # Fast JSON parsing (optional, for performance)
ujson>=5.0.0           # Ultra-fast JSON parsing (optional, for performance)

# For large file handling
dask>=2023.0.0         # Parallel computing for very large datasets
vaex>=4.0.0            # Memory-efficient data processing (optional)
```

## Output Format

### JSON Output Structure
```json
{
  "metadata": {
    "generated_at": "2025-01-27T10:00:00Z",
    "analysis_type": "comprehensive",
    "source": "file",
    "channels_analyzed": ["channel1", "channel2"],
    "processing_time": "45.2s"
  },
  "summary": {
    "total_messages": 15000,
    "total_channels": 2,
    "date_range": "2024-01-01 to 2025-01-27",
    "analysis_coverage": "100%"
  },
  "results": {
    "channel1": { ... },
    "channel2": { ... }
  }
}
```

**Key Benefits of JSON-Only Output:**
- **Consistency**: Single, well-defined output format
- **Interoperability**: Easy integration with other tools and APIs
- **Performance**: No format conversion overhead
- **Maintainability**: Simpler codebase and testing
- **Extensibility**: Easy to add new fields without breaking changes

**Pandas JSON Integration Benefits:**
- **Performance**: `pd.read_json()` is 3-5x faster than manual JSON parsing for small files
- **Memory Efficiency**: Manual chunking prevents memory issues with large files
- **Data Validation**: Automatic type inference and validation
- **Error Handling**: Robust error handling for malformed JSON
- **Scalability**: Handles files from KB to GB efficiently with appropriate strategies
- **Standardization**: Consistent JSON structure across all outputs

## Performance Optimizations

### 1. Pandas Integration Benefits
- **Efficient JSON I/O**: `pd.read_json()` for small files, manual chunking for large files
- **Memory Management**: Smart file size detection and appropriate reading strategy
- **Data Processing**: Vectorized operations for faster analysis
- **Memory Efficiency**: Better memory usage than manual JSON parsing

### 2. Parallel Processing
- **Multi-threaded data loading**
- **Parallel analysis execution**
- **Configurable worker pool size**

### 3. Caching Strategy
- **Result caching with TTL**
- **Incremental analysis support**
- **Smart cache invalidation**

### 4. Memory Management
- **Streaming data processing with pandas chunks**
- **Batch processing for large datasets**
- **Memory usage monitoring and optimization**

### 5. I/O Optimization
- **Async file operations**
- **Database connection pooling**
- **Efficient JSON serialization with pandas**
- **Smart file reading strategy (pandas for small, manual chunking for large)**

## Error Handling and Recovery

### Error Categories

1. **Data Loading Errors**
   - File not found, permission denied
   - Database connection failures
   - Network timeouts

2. **Processing Errors**
   - Invalid data format
   - Memory exhaustion
   - Processing timeouts

3. **Output Errors**
   - Disk space issues
   - Permission problems
   - Format generation failures

### Recovery Strategies

1. **Graceful Degradation**
   - Continue processing other channels
   - Return partial results
   - Provide detailed error reporting

2. **Retry Mechanisms**
   - Exponential backoff for transient failures
   - Configurable retry limits
   - Circuit breaker pattern for persistent failures

3. **User Feedback**
   - Clear error messages
   - Suggested solutions
   - Progress preservation

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock data and dependencies
- Edge case coverage

### Integration Tests
- End-to-end analysis pipeline
- Real data processing
- Performance benchmarking

### User Acceptance Tests
- CLI usability testing
- Output format validation
- Error handling verification

## Migration from Report Command

### Compatibility Layer
```python
class ReportCommandCompatibility:
    """Provides compatibility with old report command syntax."""
    
    def convert_old_options(self, old_options: Dict) -> AnalysisConfig:
        """Convert old report command options to new analysis config."""
        # Mapping logic for backward compatibility
        pass
```

### Deprecation Strategy
1. **Phase 1**: Add analysis command alongside report
2. **Phase 2**: Deprecate report command with warnings
3. **Phase 3**: Remove report command entirely

## Future Enhancements

### Phase 2 Features
- **Real-time Analysis**: Streaming data analysis
- **Custom Metrics**: User-defined analysis functions
- **API Integration**: REST API for analysis results
- **Scheduling**: Automated analysis execution

### Phase 3 Features
- **Machine Learning**: Predictive analytics
- **Data Export**: Integration with external analysis tools
- **Collaboration**: Shared analysis workspaces
- **Advanced Filtering**: Complex query language support

## Implementation Timeline

### Week 1-2: Core Architecture
- Basic command structure
- Data loading abstraction
- Configuration management

### Week 3-4: Processing Pipeline
- Analysis pipeline implementation
- Basic analyzers (messages, media)
- Output formatting

### Week 5-6: Advanced Features
- Temporal and engagement analysis
- Performance optimizations
- Error handling

### Week 7-8: Testing and Polish
- Comprehensive testing
- Documentation
- User feedback integration

## Success Criteria

1. **Functionality**: All analysis types working correctly with JSON output
2. **Performance**: Meets speed and memory targets
3. **Usability**: Intuitive CLI with helpful feedback
4. **Maintainability**: Clean, testable code structure with JSON-only output
5. **Extensibility**: Easy to add new analysis types and fields
6. **Consistency**: Single, well-defined output format across all analysis types

---

*This design specification provides the foundation for implementing a modern, efficient analysis command that significantly improves upon the previous report command.*
