# 📱 Telegram Media Messages Tool

A comprehensive, unified tool for collecting, exporting, analyzing, and managing Telegram media messages with advanced features including interactive dashboards and document analysis.

## 🚀 Features

- **📥 Message Collection**: Collect messages from Telegram channels with intelligent rate limiting
- **📤 Data Export**: Export data in multiple formats (JSON, CSV, Excel) with batch processing
- **📊 Data Analysis**: Generate comprehensive console reports and interactive HTML dashboards
- **📚 Document Analysis**: Specialized analysis of PDF/EPUB filenames for duplicates and patterns
- **📥 Data Import**: Import data from JSON/CSV files into database via API with validation and batch processing
- **⚡ Rate Limiting**: Built-in rate limiting to avoid Telegram API restrictions
- **🔄 Database Integration**: Store and retrieve message metadata from databases
- **🎨 Interactive Visualizations**: Beautiful Plotly-based HTML dashboards
- **📁 Multi-format Support**: Handle JSON and CSV input files seamlessly

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install required packages
pip install -r requirements_pandas.txt
pip install -r requirements_visualization.txt

# For interactive dashboards (optional)
pip install plotly
```

### Environment Variables

Create a `.env` file in the project directory:

```bash
# Telegram API credentials
TG_API_ID=your_api_id_here
TG_API_HASH=your_api_hash_here

# Database URL (optional)
TELEGRAM_DB_URL=http://localhost:8000
```

## 📖 Usage

### Basic Command Structure

```bash
python main.py [COMMAND] [OPTIONS]
```

### Available Commands

#### 1. 📥 **collect** - Collect Messages from Telegram Channels

Collect new messages from specified Telegram channels with rate limiting and database storage.

```bash
# Basic collection from default channels
python main.py collect

# Collect from specific channels
python main.py collect --channels "@channel1,@channel2"

# Limit messages per channel
python main.py collect --max-messages 100

# Start from specific message ID
python main.py collect --offset-id 12345

# Custom rate limiting
python main.py collect --rate-limit 60

# Dry run (no database operations)
python main.py collect --dry-run

# Verbose logging
python main.py collect --verbose
```

**Options:**
- `-c, --channels TEXT`: Comma-separated list of channel usernames
- `-m, --max-messages INTEGER`: Maximum messages to collect per channel
- `-o, --offset-id INTEGER`: Start collecting from message ID greater than this
- `-r, --rate-limit INTEGER`: Messages per minute rate limit (default: 120)
- `-s, --session-name TEXT`: Telegram session name (default: telegram_collector)
- `--dry-run`: Run without storing to database
- `-v, --verbose`: Enable verbose logging output

#### 2. 📤 **export** - Export Data from Database

Export existing Telegram message data from the database in various formats.

```bash
# Export to JSON (default)
python main.py export

# Export to CSV
python main.py export --format csv

# Export to Excel
python main.py export --format excel

# Export to all formats
python main.py export --format all

# Custom output filename
python main.py export --output my_export

# Custom batch size
python main.py export --batch-size 500

# Generate summary report
python main.py export --summary

# Verbose logging
python main.py export --verbose
```

**Options:**
- `-o, --output TEXT`: Output filename without extension (default: telegram_messages_export)
- `-f, --format [json|csv|excel|all]`: Export format (default: json)
- `-b, --batch-size INTEGER`: Batch size for fetching messages (default: 1000)
- `-s, --summary`: Generate and display summary report
- `-v, --verbose`: Enable verbose logging output

#### 3. 📊 **analyze** - Analyze Message Data

Generate comprehensive analysis reports and interactive HTML dashboards from the database.

```bash
# Basic analysis (console output only)
python main.py analyze

# Generate interactive HTML dashboard
python main.py analyze --dashboard

# Generate summary report
python main.py analyze --summary

# Both dashboard and summary with verbose logging
python main.py analyze -s -d -v
```

**Options:**
- `-d, --dashboard`: Generate interactive HTML dashboard
- `-s, --summary`: Generate summary report
- `-v, --verbose`: Enable verbose logging output

**Data Source:**
- Database analysis only (no file input required)
- Connects to FastAPI database service
- Analyzes all stored Telegram messages

#### 4. 📚 **Document Analysis** - Integrated in Main Analyze Command

Document filename analysis is now integrated into the main analyze command and provides:
- PDF and EPUB filename uniqueness analysis
- Duplicate detection and reporting
- File size distribution analysis
- MIME type breakdown

```bash
# Document analysis included in main analyze command
python main.py analyze --dashboard

# For detailed document insights, use verbose mode
python main.py analyze --dashboard --verbose
```

**Features:**
- Automatic duplicate detection
- File extension analysis
- File size categorization
- MIME type distribution

#### 5. 📥 **import** - Import Data to Database

Import Telegram message data from JSON or CSV files into the database via API.

```bash
# Basic import from JSON file
python main.py import data.json

# Import from CSV file
python main.py import data.csv

# Validate data format without importing
python main.py import data.json --validate-only

# Dry run (show what would be imported)
python main.py import data.json --dry-run

# Custom batch size for processing
python main.py import data.json --batch-size 50

# Skip duplicate messages
python main.py import data.json --skip-duplicates

# Verbose logging
python main.py import data.json --verbose
```

**Options:**
- `-b, --batch-size INTEGER`: Batch size for importing messages (default: 100)
- `--dry-run`: Show what would be imported without actually importing
- `--skip-duplicates`: Skip messages that already exist in database
- `--validate-only`: Only validate data format without importing
- `-v, --verbose`: Enable verbose logging output

**Supported Input Formats:**
- JSON files (structured with export_info, messages array, or simple array)
- CSV files with appropriate headers

**Import Process:**
1. **Data Validation**: Checks required fields (message_id, channel_username, text)
2. **Format Conversion**: Converts data to TelegramMessage objects
3. **Batch Processing**: Processes messages in configurable batches
4. **Database Storage**: Stores messages via API calls
5. **Progress Tracking**: Shows real-time import progress and statistics

## 📊 Analysis Features

### Console Reports
- **Media Type Analysis**: Distribution of different media types
- **Channel Analysis**: Message counts by channel
- **Text Length Analysis**: Statistical analysis of message text lengths
- **Temporal Pattern Analysis**: Daily and hourly message activity patterns
- **Document Analysis**: PDF/EPUB filename uniqueness and duplicate detection

### Interactive HTML Dashboards
- **Time Series Charts**: Message activity over time
- **Media Distribution**: Pie charts of media types
- **Channel Comparison**: Bar charts of messages by channel
- **Text Length Distribution**: Histograms of message lengths
- **Summary Statistics**: Comprehensive data tables
- **Responsive Design**: Mobile and desktop optimized

## 🔧 Advanced Features

### Rate Limiting
- Configurable messages per minute limits
- Automatic delays between channel switches
- Session cooldown periods
- Smart retry mechanisms

### Database Integration
- Store message metadata in databases
- Resume collection from last message ID
- Batch processing for large datasets
- Connection pooling and error handling

### Data Processing
- Automatic data type conversion
- Text cleaning and normalization
- File size analysis and statistics
- Creator and forwarding analysis

## 📁 Input File Formats

### JSON Files
The tool supports multiple JSON formats:

**Structured Format (with export_info):**
```json
{
  "export_info": {
    "exported_at": "2025-08-28T16:32:11",
    "total_messages": 1000,
    "format": "pandas_dataframe"
  },
  "data": [
    {
      "message_id": 123,
      "channel_username": "@channel",
      "text": "Message content",
      "media_type": "document",
      "file_name": "document.pdf"
    }
  ]
}
```

**Simple Array Format:**
```json
[
  {
    "message_id": 123,
    "channel_username": "@channel",
    "text": "Message content"
  }
]
```

### CSV Files
Standard CSV format with headers matching the data structure.

## 📈 Output Examples

### Console Analysis Report
```
============================================================
COMPREHENSIVE TELEGRAM DATA ANALYSIS REPORT
============================================================
Data source: telegram_messages.json
Total messages: 1,234
Date range: 2025-08-01 to 2025-08-28
Unique channels: 5
============================================================

==================================================
MEDIA TYPE ANALYSIS
==================================================
document: 567 messages (45.9%)
photo: 234 messages (19.0%)
No Media: 433 messages (35.1%)

==================================================
CHANNEL ANALYSIS
==================================================
@SherwinVakiliLibrary: 456 messages (37.0%)
@Channel2: 234 messages (19.0%)
@Channel3: 123 messages (10.0%)
```

### Document Analysis Report
```
================================================================================
📚 DOCUMENT FILENAME ANALYSIS REPORT
================================================================================

📊 SUMMARY STATISTICS:
  Total PDF/EPUB Documents: 567
  PDF Files: 523
  EPUB Files: 44
  Unique Filenames: 512
  Non-Unique Filenames: 55
  Total Unique Names: 512
  Duplicate Ratio: 9.7%

💡 RECOMMENDATIONS:
  ✅ Low duplicate ratio (9.7%) - good filename uniqueness
  🔍 Review 55 files with duplicate names
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install plotly pandas telethon aiohttp click
   ```

2. **Telegram API Errors**
   - Verify API credentials in `.env` file
   - Check rate limiting settings
   - Ensure session files are not corrupted

3. **Dashboard Generation Fails**
   - Install plotly: `pip install plotly`
   - Check file permissions for output directory
   - Verify input data format

4. **Database Connection Issues**
   - Check `TELEGRAM_DB_URL` environment variable
   - Verify database service is running
   - Check network connectivity

### Debug Mode
Enable verbose logging for detailed error information:
```bash
python main.py [COMMAND] --verbose
```

## 📚 Examples

### Complete Workflow Example

```bash
# 1. Collect messages from channels
python main.py collect --channels "@SherwinVakiliLibrary" --max-messages 100

# 2. Export collected data
python main.py export --format all --output my_collection

# 3. Analyze the data
python main.py analyze --dashboard

# 4. Analyze document filenames
python main.py analyze --dashboard --verbose

# 5. Import data to another database (optional)
python main.py import my_collection.json --validate-only
python main.py import my_collection.json --batch-size 200
```

### Batch Processing Example

```bash
# Process multiple channels with custom settings
python 0_media_messages.py collect \
  --channels "@channel1,@channel2,@channel3" \
  --max-messages 500 \
  --rate-limit 80 \
  --verbose
```

## 🔒 Security & Privacy

- **API Credentials**: Store sensitive credentials in `.env` files (not in version control)
- **Session Files**: Telegram session files contain authentication data
- **Data Privacy**: Respect user privacy when collecting and analyzing data
- **Rate Limiting**: Use appropriate rate limits to avoid API restrictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error logs
3. Enable verbose logging for detailed information
4. Check the Archive directory for specialized scripts

---

**Happy Analyzing! 🎉**

This tool provides everything you need to collect, analyze, and visualize Telegram media messages in one unified interface.
