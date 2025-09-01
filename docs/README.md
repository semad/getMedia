# Collections Directory

This directory contains Telegram channel message collections and processed data.

## Directory Structure

### `/raw/` - Original Collection Files
Contains the original, unprocessed collection files from Telegram channels:
- `tg_{channel}_{start_id}_{end_id}.json` - Individual collection files
- `tg_{channel}_{start_id}_{end_id}_combined.json` - Combined collection files

### Root Level - Processed Files
Contains processed, combined, and analysis-ready files:
- Combined collections (created using `combine` command)
- Analysis reports
- Processed datasets

## File Naming Convention

### Individual Collections
```
tg_{channel_name}_{start_message_id}_{end_message_id}.json
```

**Examples:**
- `tg_SherwinVakiliLibrary_1_150244.json` - Messages 1-150244
- `tg_books_1_482.json` - Messages 1-482
- `tg_Free_Books_life_1_118.json` - Messages 1-118

### Combined Collections
```
tg_{channel_name}_{overall_start}_{overall_end}_combined.json
```

**Examples:**
- `tg_SherwinVakiliLibrary_1_150300_combined.json` - All messages 1-150300

## Usage

### Combine Collections
```bash
# Combine all available channels (auto-detect)
python main.py combine

# Combine specific channels
python main.py combine -c "SherwinVakiliLibrary,books"

# Combine with verbose logging
python main.py combine -v

# Combine specific channels with verbose logging
python main.py combine -c "SherwinVakiliLibrary,books" -v
```

### Import Collections
```bash
# Import a combined collection
python main.py import reports/collections/raw/tg_SherwinVakiliLibrary_1_150300_combined.json
```

## Data Structure

Each collection file contains:
- **metadata**: Collection information, timestamps, channel details
- **messages**: Array of Telegram messages with full content
- **message_id**: Unique identifier for each message
- **channel_username**: Source channel name
- **date**: Message timestamp
- **text**: Message content
- **media_type**: Type of media (if any)
- **file_name**: Associated filename (if media)

## File Sizes

- **Small collections**: 50-100 KB (100-200 messages)
- **Medium collections**: 200-500 KB (500-1000 messages)  
- **Large collections**: 50-100 MB (100,000+ messages)
- **Combined files**: Varies based on total message count

## Best Practices

1. **Keep original files** in `/raw/` for backup
2. **Use combined files** for analysis and processing
3. **Regular cleanup** of duplicate combined files
4. **Monitor disk space** for large collections
5. **Backup important collections** before processing
6. **New collections** automatically go to `/raw/` directory
7. **Combined files** are created in main directory for easy access

## Processing Workflow

1. **Collect** → New collection files automatically saved to `/raw/`
2. **Combine** → Create consolidated files in main directory
3. **Analyze** → Generate reports and insights
4. **Import** → Load into database for further processing
5. **Archive** → Move processed files to appropriate locations

## Collection Behavior

### New Collections (collect command)
- **Default location**: `./reports/collections/raw/`
- **File naming**: `tg_{channel}_{start_id}_{end_id}.json`
- **Purpose**: Store original, unprocessed collection data

### Combined Collections (combine command)
- **Location**: `./reports/collections/` (main directory)
- **File naming**: `tg_{channel}_{overall_start}_{overall_end}_combined.json`
- **Purpose**: Processed, consolidated data ready for analysis
