# Telegram Media Messages Tool - Modular Structure

This document describes the new modular architecture of the Telegram Media Messages Tool, which has been refactored from a single large file into a clean, maintainable package structure.

## 🏗️ Architecture Overview

The application has been broken down into logical modules organized in a `modules/` directory, with a single entry point:

```
getMedia/
├── __init__.py              # Package initialization and exports
├── main.py                  # 🆕 Main entry point and CLI (combined)
├── modules/                 # Core modules directory
│   ├── __init__.py         # Modules package initialization
│   ├── models.py            # Data models and dataclasses
│   ├── retry_handler.py     # Retry logic with exponential backoff
│   ├── database_service.py  # Database operations and API communication
│   ├── import_processor.py  # Import logic and data processing
│   ├── telegram_collector.py # 🆕 Telegram message collection
│   ├── telegram_exporter.py # 🆕 Data export functionality
│   └── telegram_analyzer.py # 🆕 Data analysis and reporting
├── requirements.txt         # Python dependencies
└── README_MODULAR.md        # This documentation
```

## 📦 Module Descriptions

### 1. **`modules/models.py`** - Data Models
Contains all the dataclasses and data structures:
- `TelegramMessage` - Core message data structure
- `ChannelConfig` - Channel configuration settings
- `RateLimitConfig` - Rate limiting parameters
- `ImportStats` - Import operation statistics

### 2. **`modules/retry_handler.py`** - Retry Logic
Implements exponential backoff for failed operations:
- Configurable retry attempts and delays
- Automatic retry for transient failures
- Logging of retry attempts and success

### 3. **`modules/database_service.py`** - Database Operations
Handles all database interactions:
- Connection management with async context manager
- Data cleaning for JSON serialization
- API communication with the database service
- Connection health checks

### 4. **`modules/import_processor.py`** - Import Logic
Core import and validation functionality:
- File loading and validation
- Data quality checking
- Batch processing
- Progress tracking and statistics
- Unified import/validation pipeline

### 5. **`modules/telegram_collector.py`** - Message Collection 🆕
Telegram message collection functionality:
- Collect messages from Telegram channels
- Rate limiting and session management
- Message processing and metadata extraction
- Database integration for storage

### 6. **`modules/telegram_exporter.py`** - Data Export 🆕
Export functionality for various formats:
- Export to JSON, CSV, and Excel formats
- Batch processing for large datasets
- Summary report generation
- Memory-efficient data handling

### 7. **`modules/telegram_analyzer.py`** - Data Analysis 🆕
Comprehensive data analysis and reporting:
- Channel analysis and statistics
- Media type breakdown and file size analysis
- Temporal analysis and patterns
- Interactive HTML dashboard support

### 8. **`main.py`** - Entry Point and CLI 🆕
**Combined entry point and command-line interface** using Click framework:
- `import` - Import, validate, and check quality of messages 🆕
- `collect` - Collect messages from Telegram channels 🆕
- `export` - Export messages to various formats 🆕
- `analyze` - Analyze data and generate reports 🆕
- `stats` - Show database statistics
- All CLI functionality in a single file

## 🚀 Usage

### Basic Commands

```bash
# Show help
python main.py --help

# Import data with limit
python main.py import reports/exports/current_db_export.json --limit 1000

# Validate data quality
python main.py import reports/exports/current_db_export.json --validate-only --check-quality

# Dry run import
python main.py import reports/exports/current_db_export.json --dry-run --limit 100

# Collect messages from channels
python main.py collect --channels "@SherwinVakiliLibrary,@books_magazine" --max-messages 100

# Export data to various formats
python main.py export --format all --summary

# Analyze data and generate reports
python main.py analyze --verbose
```

### Import Options

```bash
python main.py import <file> [OPTIONS]
  -v, --verbose             Enable verbose logging
  -b, --batch-size INT     Batch size (default: 100)
  --dry-run                Show what would be imported
  --validate-only          Only validate without importing
  --check-quality          Check data quality
  --limit INT              Limit number of records
  --max-retries INT        Maximum retry attempts (default: 5)
  --retry-delay FLOAT      Base retry delay (default: 1.0)
  --batch-delay FLOAT      Delay between batches (default: 1.0)
```

### Collect Options

```bash
python main.py collect [OPTIONS]
  -c, --channels TEXT      Comma-separated list of channel usernames
  -m, --max-messages INT   Maximum messages to collect per channel
  -o, --offset-id INT      Start collecting from message ID greater than this
  -r, --rate-limit INT     Messages per minute rate limit (default: 120)
  -s, --session-name TEXT  Telegram session name
  --dry-run                Run without storing to database
  -v, --verbose            Enable verbose logging output
```

### Export Options

```bash
python main.py export [OPTIONS]
  -o, --output TEXT        Output filename without extension
  -f, --format CHOICE      Export format: json, csv, excel, or all
  -b, --batch-size INT     Batch size for fetching messages (default: 1000)
  -v, --verbose            Enable verbose logging output
  -s, --summary            Generate and display summary report
```

### Analyze Options

```bash
python main.py analyze [OPTIONS]
  -v, --verbose            Enable verbose logging output
  -d, --dashboard          Generate interactive HTML dashboard
  -s, --summary            Generate summary report
```

## 🔧 Configuration

### Environment Variables
The following environment variables are required for the collect command:
- `TG_API_ID` - Your Telegram API ID
- `TG_API_HASH` - Your Telegram API Hash
- `TELEGRAM_DB_URL` - Database service URL (default: http://localhost:8000)

### Database URL
The database URL is currently hardcoded in `main.py` as `http://localhost:8000`. You can modify this or make it configurable through environment variables.

### Batch Processing
- Default batch size: 100 messages (import), 1000 messages (export)
- Default batch delay: 1 second
- Configurable retry parameters

## 📊 Features

### Data Quality Checking
- Automatic detection of `NaN` and `inf` values
- Validation of required fields
- Reporting of data quality issues

### Error Handling
- Exponential backoff for failed operations
- Connection health checks
- Comprehensive error logging

### Performance
- Batch processing with configurable delays
- Progress tracking and statistics
- Memory-efficient processing of large datasets

### Message Collection 🆕
- Telegram channel message collection
- Rate limiting and session management
- Automatic message ID tracking
- Media file metadata extraction

### Data Export 🆕
- Multiple export formats (JSON, CSV, Excel)
- Batch processing for large datasets
- Summary report generation
- Memory usage optimization

### Data Analysis 🆕
- Comprehensive channel analysis
- Media type and file size statistics
- Temporal pattern analysis
- Interactive dashboard support

## 🛠️ Development

### Adding New Features
1. **New Models**: Add to `modules/models.py`
2. **New Services**: Create new service modules in `modules/`
3. **New Commands**: Add to `main.py` (CLI section)
4. **New Processors**: Extend `modules/import_processor.py`
5. **New Collectors**: Extend `modules/telegram_collector.py`
6. **New Exporters**: Extend `modules/telegram_exporter.py`
7. **New Analyzers**: Extend `modules/telegram_analyzer.py`

### Testing
```bash
# Test validation
python main.py import reports/exports/current_db_export.json --validate-only --limit 10

# Test dry run import
python main.py import reports/exports/current_db_export.json --dry-run --limit 10

# Test with verbose logging
python main.py import reports/exports/current_db_export.json --verbose --limit 100

# Test data analysis
python main.py analyze --verbose

# Test export (requires database connection)
python main.py export --format json --summary
```

## 📋 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- `click>=8.0.0` - Command line interface
- `aiohttp>=3.8.0` - Async HTTP client
- `telethon>=1.28.0` - Telegram client library 🆕
- `pandas>=1.5.0` - Data analysis and manipulation 🆕
- `openpyxl>=3.0.0` - Excel file support 🆕
- `asyncio` - Async support (built-in)
- `pathlib` - Path handling (built-in)

## 🔄 Migration from Original Script

The original `0_media_messages.py` script has been completely refactored into this modular structure. **All functionality has been preserved and enhanced**:

- ✅ All original features maintained
- ✅ Enhanced error handling and retry logic
- ✅ Better data quality checking
- ✅ Cleaner, more maintainable code
- ✅ Easier to extend and modify
- ✅ Better separation of concerns
- ✅ Professional package structure with modules directory
- ✅ **Simplified structure with combined main.py** 🆕
- ✅ **Complete collect functionality restored** 🆕
- ✅ **Complete export functionality restored** 🆕
- ✅ **Complete analyze functionality restored** 🆕

## 🎯 Benefits of Modular Structure

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Reusability**: Modules can be imported and used independently
4. **Extensibility**: Easy to add new features without modifying existing code
5. **Readability**: Smaller, focused files are easier to understand
6. **Collaboration**: Multiple developers can work on different modules
7. **Organization**: Clear separation with modules directory
8. **Professional**: Follows Python packaging best practices
9. **Simplified**: Single entry point with all CLI functionality 🆕
10. **Complete**: All original functionality restored and enhanced 🆕

## 🚨 Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the `getMedia` directory:
```bash
cd getMedia
python main.py --help
```

### Missing Dependencies
Install required packages:
```bash
pip install click aiohttp telethon pandas openpyxl
```

### Telegram API Setup
For the collect command, you need to set up Telegram API credentials:
```bash
export TG_API_ID="your_api_id"
export TG_API_HASH="your_api_hash"
```

### Database Connection Issues
- Check if the database service is running on `localhost:8000`
- Verify network connectivity
- Check firewall settings

## 📈 Future Enhancements

Potential areas for future development:
- Configuration file support
- Database connection pooling
- More data format support (CSV, XML)
- Advanced filtering and querying
- Performance monitoring and metrics
- Plugin system for custom processors
- Additional modules for different functionality
- Environment variable configuration
- Interactive HTML dashboard generation
- Real-time data streaming
- Advanced media processing
