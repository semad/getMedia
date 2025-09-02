# Telegram Media Messages Tool

A powerful CLI tool for collecting, processing, analyzing, and visualizing Telegram messages from channels.

## 🚀 Quick Start

```bash
# 1. Collect messages from a channel
python main.py collect -c "channel_name" -m 100

# 2. Combine existing collection files
python main.py combine

# 3. Import to database
python main.py import reports/collections/tg_channel_combined.json

# 4. Generate reports
python main.py report messages

# 5. Run advanced analysis
python main.py analysis --channels "books" --verbose

# 6. Create interactive dashboard
python main.py dashboard

# Get help for any command
python main.py --help
```

## 📋 Commands

### 1. Collect Command

Collects messages from Telegram channels and exports them to organized JSON files.

#### Basic Usage
```bash
python main.py collect [OPTIONS]
```

#### Options
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--channels` | `-c` | Comma-separated list of channel usernames | `@SherwinVakiliLibrary` |
| `--max-messages` | `-m` | Maximum messages to collect per channel | No limit |
| `--offset-id` | `-o` | Start collecting from message ID greater than this | 0 |
| `--rate-limit` | `-r` | Messages per minute rate limit | 120 |
| `--session-name` | `-s` | Telegram session name | `telegram_collector` |
| `--file-name` | `-f` | Custom export filename | Auto-generated |
| `--verbose` | `-v` | Enable verbose logging output | False |
| `--help` | `-h` | Show help message | - |

#### Examples

**Collect 100 messages from a specific channel:**
```bash
python main.py collect -c "books" -m 100 -v
```

**Collect from multiple channels:**
```bash
python main.py collect -c "SherwinVakiliLibrary,books,Free_Books_life" -m 500
```

**Collect with custom rate limiting:**
```bash
python main.py collect -c "books" -m 1000 -r 60 -v
```

**Collect with custom session:**
```bash
python main.py collect -c "books" -s "my_session" -v
```

**Collect with offset (continue from specific message ID):**
```bash
python main.py collect -c "books" -o 1000 -m 500
```

### 2. Combine Command

Combines existing collection files for the same channel into consolidated files.

#### Basic Usage
```bash
python main.py combine [OPTIONS]
```

#### Options
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--channels` | `-c` | Comma-separated list of channel usernames | Auto-detect |
| `--verbose` | `-v` | Enable verbose logging output | False |
| `--help` | `-h` | Show help message | - |

#### Examples

**Auto-detect and combine all available channels:**
```bash
python main.py combine
```

**Combine specific channels:**
```bash
python main.py combine -c "SherwinVakiliLibrary,books"
```

**Combine with verbose logging:**
```bash
python main.py combine -v
```

**Combine specific channels with verbose logging:**
```bash
python main.py combine -c "SherwinVakiliLibrary,books" -v
```

### 3. Import Command

Imports collected messages from JSON files into the database for analysis and processing.

#### Basic Usage
```bash
python main.py import [OPTIONS] IMPORT_FILE
```

#### Examples

**Import a combined collection file:**
```bash
python main.py import reports/collections/tg_books_1_482_combined.json
```

**Import with verbose logging:**
```bash
python main.py import reports/collections/tg_books_1_482_combined.json -v
```

### 4. Report Command

Generates various types of reports for Telegram data analysis.

#### Basic Usage
```bash
python main.py report [OPTIONS] REPORT_TYPE
```

#### Report Types
- `messages` - Generate message analysis reports
- `channels` - Create channel summary reports
- `stats` - Generate statistical reports

#### Examples

**Generate message reports:**
```bash
python main.py report messages
```

**Generate channel reports:**
```bash
python main.py report channels
```

### 5. Analysis Command

Performs comprehensive analysis of Telegram channel data including filename analysis, filesize analysis, and message content analysis with advanced pattern recognition and language detection.

#### Basic Usage
```bash
python main.py analysis [OPTIONS]
```

#### Options
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--channels` | `-c` | Comma-separated list of channel usernames | All available channels |
| `--output-dir` | `-o` | Output directory for analysis results | `reports/analysis` |
| `--enable-file-source` | `-f` | Enable file-based data source | `True` |
| `--enable-api-source` | `-a` | Enable API-based data source | `True` |
| `--enable-diff-analysis` | `-d` | Enable diff analysis between sources | `True` |
| `--verbose` | `-v` | Enable verbose logging output | `False` |
| `--help` | `-h` | Show help message | - |

#### Analysis Types

**Filename Analysis:**
- Duplicate filename detection
- Filename pattern analysis (length, extensions, special characters)
- Filename quality assessment

**Filesize Analysis:**
- Duplicate filesize detection
- Filesize distribution analysis with size bins
- Potential duplicate file identification

**Message Analysis:**
- Content statistics and engagement metrics
- Pattern recognition (hashtags, mentions, URLs, emojis)
- Creator analysis and activity patterns
- Language detection using character frequency analysis

#### Examples

**Run analysis on specific channels:**
```bash
python main.py analysis -c "books,SherwinVakiliLibrary" -v
```

**Run analysis with custom output directory:**
```bash
python main.py analysis -c "books" -o "custom_analysis" -v
```

**Run analysis with only file source:**
```bash
python main.py analysis -c "books" --enable-api-source false -v
```

**Run analysis with diff analysis disabled:**
```bash
python main.py analysis -c "books" --enable-diff-analysis false -v
```

**Run analysis on all available channels:**
```bash
python main.py analysis -v
```

#### Output Structure

Analysis results are saved as JSON files with the following structure:

```json
{
  "source": "file",
  "channel_name": "@books",
  "analysis_timestamp": "2025-01-15T10:30:00",
  "total_records_analyzed": 1000,
  "filename_analysis": {
    "duplicate_filename_detection": {
      "total_files": 500,
      "total_unique_filenames": 450,
      "files_with_duplicate_names": 50,
      "duplicate_ratio": 0.11,
      "most_common_filenames": {...}
    },
    "filename_pattern_analysis": {
      "length_statistics": {...},
      "extension_distribution": {...},
      "special_character_analysis": {...}
    }
  },
  "filesize_analysis": {
    "duplicate_filesize_detection": {...},
    "filesize_distribution_analysis": {...}
  },
  "message_analysis": {
    "content_statistics": {...},
    "pattern_recognition": {...},
    "creator_analysis": {...},
    "language_analysis": {...}
  },
  "performance_metrics": {
    "elapsed_time": 15.2,
    "memory_usage_mb": 150.5
  }
}
```

### 6. Dashboard Command

Creates interactive Plotly dashboards from channel reports and data.

#### Basic Usage
```bash
python main.py dashboard [OPTIONS]
```

#### Examples

**Generate interactive dashboard:**
```bash
python main.py dashboard
```

**Generate dashboard with verbose logging:**
```bash
python main.py dashboard -v
```

## 📁 File Structure

### Directory Organization
```
reports/
├── collections/
│   ├── raw/                           # Individual collection files
│   │   ├── tg_books_1_482.json       # Messages 1-482
│   │   ├── tg_books_472_482.json     # Messages 472-482
│   │   └── tg_SherwinVakiliLibrary_1_150244.json
│   └──                               # Combined files (created by combine command)
│       ├── tg_books_1_482_combined.json
│       └── tg_SherwinVakiliLibrary_1_150300_combined.json
├── channels/                          # Channel reports and summaries
├── html/                             # Interactive dashboards and HTML reports
└── analysis/                         # Advanced analysis results and reports
    ├── filename_analysis/            # Filename analysis results
    ├── filesize_analysis/            # Filesize analysis results
    ├── message_analysis/             # Message content analysis results
    └── combined_analysis/            # Combined analysis reports
```

### File Naming Convention

#### Individual Collections
```
tg_{channel_name}_{start_message_id}_{end_message_id}.json
```

**Examples:**
- `tg_books_1_482.json` - Messages 1-482 from books channel
- `tg_SherwinVakiliLibrary_1_150244.json` - Messages 1-150244 from SherwinVakiliLibrary
- `tg_Free_Books_life_1_118.json` - Messages 1-118 from Free_Books_life

#### Combined Collections
```
tg_{channel_name}_{overall_start}_{overall_end}_combined.json
```

**Examples:**
- `tg_books_1_482_combined.json` - All messages 1-482 combined
- `tg_SherwinVakiliLibrary_1_150300_combined.json` - All messages 1-150300 combined

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Telegram API credentials
export TG_API_ID=your_api_id
export TG_API_HASH=your_api_hash
export TG_SESSION=your_session_name
export TG_username=your_username
export TG_phone=your_phone_number

# Database configuration
export DB_HOST=localhost
export DB_NAME=aiLib
export DB_USER=username
export PGPASSWORD=password
export DB_PORT=5432
```

### Rate Limiting
- **Default**: 120 messages per minute
- **Customizable**: Use `-r` flag to set custom rate
- **Recommended**: 60-120 for stable connections

## 📊 Data Structure

Each collection file contains:

```json
{
  "metadata": {
    "collected_at": "2025-08-31T13:07:42",
    "channels": ["books"],
    "data_format": "structured_dataframe",
    "total_messages": 5,
    "dataframe_info": {
      "shape": [5, 19],
      "columns": ["message_id", "channel_username", "date", "text", ...],
      "dtypes": {...}
    }
  },
  "messages": [
    {
      "message_id": 472,
      "channel_username": "books",
      "date": "2025-08-31T10:00:00",
      "text": "Message content...",
      "media_type": null,
      "views": 150,
      "forwards": 5,
      "replies": 2
    }
  ]
}
```

## 🚀 Complete Workflow

### Phase 1: Collection
```bash
# Collect messages from channels
python main.py collect -c "books" -m 1000 -v

# Files are automatically saved to reports/collections/raw/
# with proper naming: tg_books_1_1000.json
```

### Phase 2: Combination
```bash
# Combine all collection files for a channel
python main.py combine -c "books"

# Creates consolidated file: reports/collections/tg_books_1_1000_combined.json
```

### Phase 3: Import
```bash
# Import combined data into database
python main.py import reports/collections/tg_books_1_1000_combined.json

# Data is now available for analysis and reporting
```

### Phase 4: Reporting
```bash
# Generate various analysis reports
python main.py report messages
python main.py report channels
python main.py report stats

# Reports are generated and saved to appropriate directories
```

### Phase 5: Advanced Analysis
```bash
# Run comprehensive analysis on channel data
python main.py analysis -c "books" -v

# Analysis results are saved to reports/analysis/ with detailed JSON reports
```

### Phase 6: Dashboard
```bash
# Create interactive visualizations
python main.py dashboard

# Interactive HTML dashboards are created in reports/html/
```

## 💡 Best Practices

### Collection Phase
- **Start small**: Begin with 100-500 messages to test
- **Use rate limiting**: Avoid overwhelming the API
- **Monitor progress**: Use `-v` flag for detailed logging
- **Save frequently**: Collections are automatically saved to raw directory

### Combination Phase
- **Auto-detect**: Use `python main.py combine` to find all channels
- **Specific channels**: Use `-c` flag for targeted combination
- **Regular cleanup**: Combine files periodically to maintain organization

### Import Phase
- **Use combined files**: Import consolidated data for efficiency
- **Validate data**: Ensure JSON files are properly formatted
- **Monitor progress**: Check import logs for any issues

### Reporting Phase
- **Multiple report types**: Generate different types of analysis
- **Review outputs**: Check generated reports for insights
- **Organize results**: Keep reports organized by date/channel

### Analysis Phase
- **Start with specific channels**: Use `-c` flag to analyze specific channels first
- **Use verbose logging**: Enable `-v` flag to monitor analysis progress
- **Review analysis results**: Check JSON outputs for insights and patterns
- **Compare sources**: Use diff analysis to compare file vs API data
- **Monitor performance**: Check memory usage and execution time metrics

### Dashboard Phase
- **Interactive analysis**: Use dashboards for data exploration
- **Share insights**: HTML dashboards can be shared with team members
- **Regular updates**: Regenerate dashboards after new data imports

### File Management
- **Keep raw files**: Original collections serve as backups
- **Use combined files**: For import and analysis
- **Monitor disk space**: Large collections can be several GB
- **Backup important data**: Before major operations

## 🐛 Troubleshooting

### Common Issues

**Missing API credentials:**
```bash
ERROR: Missing TG_API_ID or TG_API_HASH environment variables
```
**Solution**: Ensure `.env` file exists and is sourced properly.

**Rate limiting errors:**
```bash
# Reduce rate limit
python main.py collect -c "channel" -r 30
```

**Connection issues:**
```bash
# Use custom session
python main.py collect -c "channel" -s "new_session"
```

**Import errors:**
```bash
# Check file format and database connection
python main.py import file.json -v
```

### Getting Help

```bash
# Command help
python main.py --help

# Specific command help
python main.py collect --help
python main.py combine --help
python main.py import --help
python main.py report --help
python main.py analysis --help
python main.py dashboard --help

# Verbose logging for debugging
python main.py collect -c "channel" -v
```

## 📈 Performance Tips

- **Batch collection**: Collect in smaller batches (100-1000 messages)
- **Efficient combining**: Use auto-detection for multiple channels
- **Rate optimization**: Balance between speed and stability
- **Session reuse**: Use consistent session names for faster connections
- **Database optimization**: Use appropriate indexes for large datasets

## 🔄 Updates and Maintenance

- **Regular collections**: Update data periodically
- **File cleanup**: Remove old combined files when no longer needed
- **Backup strategy**: Maintain copies of important collections
- **Monitor changes**: Track channel activity and adjust collection frequency
- **Dashboard updates**: Regenerate visualizations after new data imports

## 📊 Output Examples

### Collection Output
```
📁 Channel books: 1000 messages → reports/collections/raw/tg_books_1_1000.json
🎯 Collection completed!
📊 Total messages collected: 1000
📺 Channels processed: 1
```

### Combine Output
```
✅ Combined file created: tg_books_1_1000_combined.json
📊 Total messages: 1000
📁 Source files: 2
🔢 Message ID range: 1-1000
```

### Import Output
```
🔄 Importing messages from existing file: tg_books_1_1000_combined.json
✅ Successfully imported 1000 messages to database
```

### Report Output
```
📊 Generating message analysis reports...
📁 Reports saved to: reports/channels/
✅ Report generation completed!
```

### Analysis Output
```
🔍 Starting advanced analysis for channel: @books
📊 Loading data from file source...
📈 Running filename analysis...
📈 Running filesize analysis...
📈 Running message analysis...
📁 Analysis results saved to: reports/analysis/
✅ Analysis completed successfully!
📊 Total records analyzed: 1000
⏱️  Execution time: 15.2 seconds
💾 Memory usage: 150.5 MB
```

### Dashboard Output
```
🎨 Creating interactive dashboard...
📊 Dashboard generated successfully!
📁 Location: reports/html/dashboard.html
```

---

For more information, run `python main.py --help` or check the individual command help with `python main.py [command] --help`.
