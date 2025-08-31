# Telegram Media Library

A comprehensive tool for collecting, analyzing, and managing Telegram messages and media files.

## Features

- **Message Collection**: Collect messages from Telegram channels with rate limiting
- **Data Analysis**: Comprehensive analysis with field-by-field insights
- **Interactive Dashboards**: Beautiful HTML dashboards with Plotly charts
- **Multi-channel Support**: Handle multiple channels simultaneously
- **Structured Data Export**: DataFrame-compatible JSON format
- **Database Integration**: Import collected data into databases

## Installation

### Prerequisites
- Python 3.11+
- `uv` package manager (install with: `pip install uv`)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd getMedia

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install click aiohttp telethon pandas numpy openpyxl plotly
```

## Dependencies

All dependencies are consolidated in a single installation:

- **Core**: `click`, `aiohttp`, `telethon`
- **Data Processing**: `pandas`, `numpy`
- **File Handling**: `openpyxl`
- **Visualization**: `plotly`

## Directory Structure

```
getMedia/
├── reports/
│   ├── collections/          # Collected JSON data files
│   ├── analysis/            # Analysis reports and logs
│   ├── html/                # All HTML dashboards and reports
│   ├── exports/             # Exported data files
│   ├── logs/                # Application logs
│   └── telegram/            # Telegram-specific reports
├── modules/                  # Core analysis modules
├── config.py                 # Configuration and constants
└── main.py                   # Main CLI application
```

## File Organization

- **HTML Files**: All dashboards and HTML reports are automatically saved to `./reports/html/`
- **Data Files**: Collections and exports are stored in `./reports/collections/`
- **Analysis**: Text-based analysis reports are generated in the console
- **Index**: Navigate to `./reports/html/index.html` to access all reports

## Usage

### Collect Messages
```bash
python main.py collect @channel_name --max-messages 1000
```

### Analyze Data
```bash
python main.py analyze --dashboard
```

### Import to Database
```bash
python main.py import --file data.json
```

## Requirements

- Python 3.11+
- Telegram API credentials (TG_API_ID, TG_API_HASH)
- Internet connection for Telegram API access
- `.env` file with your Telegram credentials

## Environment Setup

Create a `.env` file in the project root:
```env
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
```
