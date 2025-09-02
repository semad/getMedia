# Report Command Documentation

## Overview

The `report` command is a powerful analysis tool that generates comprehensive reports from Telegram message data. It can process data from two sources:
1. **File-based analysis** - from combined JSON collection files
2. **Database analysis** - from database API endpoints

## Command Syntax

```bash
python main.py report [OPTIONS]
```

## Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--file-messages` | `-f` | Generate message analysis reports from combined files | `False` |
| `--db-messages` | `-d` | Generate message analysis reports from database API endpoints | `False` |
| `--summary` | `-s` | Generate summary files only (JSON + text) | `False` |
| `--verbose` | `-v` | Enable verbose logging output | `False` |
| `--help` | `-h` | Show help message | - |

## Usage Examples

### File-based Analysis
```bash
# Generate reports from combined JSON files
python main.py report --file-messages

# With verbose logging
python main.py report --file-messages --verbose
```

### Database Analysis
```bash
# Generate reports from database API endpoints
python main.py report --db-messages

# With verbose logging
python main.py report --db-messages --verbose
```

### Summary Generation
```bash
# Generate summary files only
python main.py report --file-messages --summary
python main.py report --db-messages --summary
```

## How It Works

### 1. File-based Analysis (`--file-messages`)

**Data Source**: Combined JSON files in `reports/collections/` directory
**Output**: Reports saved to `reports/analysis/file_messages/channels/` directory

**Process Flow**:
1. **File Discovery**: Scans for files matching pattern `*_combined.json`
2. **JSON Parsing**: Uses multiple fallback strategies to parse JSON data
3. **Data Conversion**: Converts messages to pandas DataFrame
4. **Analysis**: Generates comprehensive reports using pandas analysis
5. **File Generation**: Saves reports in JSON and text formats

**Generated Files**:
- `{channel}_report.json` - Comprehensive analysis report
- `{channel}_summary.txt` - Human-readable summary

### 2. Database Analysis (`--db-messages`)

**Data Source**: Database API endpoints (default: `http://localhost:8000`)
**Output**: Reports saved to `reports/analysis/db_messages/channels/` directory

**Process Flow**:
1. **API Connection**: Establishes connection to database API
2. **Channel Discovery**: Auto-detects available channels or uses specified list
3. **Data Fetching**: Retrieves messages via REST API calls
4. **Data Processing**: Converts to pandas DataFrame for analysis
5. **Report Generation**: Creates comprehensive analysis reports
6. **File Saving**: Saves reports to configured output directory

**API Endpoints Used**:
- `GET /api/v1/telegram/channels` - List available channels
- `GET /api/v1/telegram/channels/{channel}/messages` - Fetch channel messages

## Report Content

### Core Analysis Components

1. **Basic Statistics**
   - Total message count
   - Dataframe dimensions
   - Column count and types

2. **Field Analysis**
   - Data type analysis
   - Null value statistics
   - Unique value counts
   - Field completeness scores

3. **Media Analysis**
   - Media message count
   - Text message count
   - Media vs text percentages

4. **Date Analysis**
   - Date range coverage
   - Active days count
   - Temporal patterns

5. **File Size Analysis**
   - Total file sizes
   - Average file sizes
   - Size distribution

6. **Engagement Analysis**
   - Views, likes, comments
   - Engagement metrics
   - Interaction patterns

### Report Structure

```json
{
  "channel_name": "channel_name",
  "generated_at": "2025-09-01T17:00:00",
  "total_messages": 1000,
  "total_columns": 15,
  "dataframe_shape": "1000 rows x 15 columns",
  "media_messages": 150,
  "text_messages": 850,
  "total_file_size": 1048576,
  "date_range": "2024-01-01 to 2025-09-01",
  "active_days": 365,
  "field_analysis": { ... },
  "media_analysis": { ... },
  "date_analysis": { ... },
  "file_analysis": { ... },
  "engagement_analysis": { ... }
}
```

## Configuration

### Directory Structure

```
reports/
├── collections/                    # Input: Combined JSON files
│   ├── tg_channel1_combined.json
│   └── tg_channel2_combined.json
├── messages/                       # Output: File-based reports
│   ├── channel1_report.json
│   ├── channel1_summary.txt
│   ├── channel2_report.json
│   └── channel2_summary.txt
└── analysis/
    └── db_messages/
        └── channels/              # Output: Database reports
            ├── channel1_report.json
            └── channel2_report.json
```

### Configuration Constants

```python
# From config.py
COLLECTIONS_DIR = "reports/collections"
FILES_CHANNELS_DIR = "reports/analysis/file_messages/channels"
DB_CHANNELS_DIR = "reports/analysis/db_messages/channels"
DEFAULT_DB_URL = "http://localhost:8000"
```

## Error Handling

### Common Issues

1. **No Data Found**
   - Check if combined JSON files exist in `reports/collections/`
   - Verify file naming follows pattern `*_combined.json`

2. **Database Connection Issues**
   - Ensure database API is running on configured URL
   - Check network connectivity and firewall settings

3. **JSON Parsing Errors**
   - Verify JSON file format and structure
   - Check for corrupted or malformed files

4. **Permission Issues**
   - Ensure write permissions for output directories
   - Check disk space availability

### Error Recovery

- **Graceful Degradation**: Continues processing other files if one fails
- **Detailed Logging**: Provides comprehensive error information
- **Partial Results**: Returns partial results even if some processing fails

## Performance Considerations

### File Processing
- **Batch Processing**: Processes multiple files sequentially
- **Memory Management**: Uses pandas for efficient data handling
- **Error Isolation**: Individual file failures don't affect others

### Database Processing
- **Async Operations**: Uses asyncio for non-blocking API calls
- **Connection Pooling**: Efficient HTTP session management
- **Rate Limiting**: Respects API rate limits and timeouts

## Integration Points

### Input Dependencies
- **Collect Command**: Generates raw message files
- **Combine Command**: Creates combined JSON collections
- **Import Command**: Populates database for API analysis

### Output Usage
- **Dashboard Generation**: Reports feed into visualization dashboards
- **Data Export**: JSON reports can be processed by external tools
- **Analysis Pipeline**: Summary files support further analysis workflows

## Best Practices

### File Organization
1. Use consistent naming conventions for input files
2. Organize output by analysis type and date
3. Maintain clear separation between raw and processed data

### Error Monitoring
1. Enable verbose logging for debugging
2. Monitor error counts and failure rates
3. Set up alerts for critical processing failures

### Performance Optimization
1. Process large datasets during off-peak hours
2. Use appropriate batch sizes for memory management
3. Monitor API rate limits and adjust accordingly

## Troubleshooting

### Debug Mode
```bash
# Enable verbose logging for detailed information
python main.py report --file-messages --verbose
```

### Common Commands
```bash
# Check available files
ls -la reports/collections/

# Verify output directories
ls -la reports/messages/
ls -la reports/analysis/db_messages/channels/

# Test database connectivity
curl http://localhost:8000/api/v1/telegram/channels
```

### Log Analysis
- Check timestamp patterns for processing duration
- Monitor success/failure ratios
- Review error messages for root cause analysis

## Future Enhancements

### Planned Features
1. **Parallel Processing**: Multi-threaded file processing
2. **Incremental Updates**: Delta processing for new data
3. **Real-time Monitoring**: Live progress tracking
4. **Custom Report Templates**: Configurable output formats
5. **Data Validation**: Schema validation and quality checks

### API Improvements
1. **Pagination Support**: Handle large datasets efficiently
2. **Caching Layer**: Reduce redundant API calls
3. **Authentication**: Secure API access controls
4. **Rate Limiting**: Intelligent request throttling

---

*This documentation covers the current implementation as of commit `fe4b61f`. For the latest updates, refer to the git history and code comments.*
