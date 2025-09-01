# File Organization Design Document

## Overview
This document describes the current file organization structure of the Telegram Media Messages Tool. The project follows a **configuration-driven architecture** with all file paths, naming patterns, and directory structures defined in `config.py`. This eliminates hardcoded paths throughout the codebase and provides a centralized, maintainable approach to file organization.

## Project Root Structure
```
getMedia/
├── main.py                 # Main CLI application entry point
├── config.py               # Centralized configuration constants
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project metadata and tool configuration
├── README.md              # Project documentation
├── modules/               # Core functionality modules
├── templates/             # HTML templates for dashboards
├── reports/               # All generated outputs and data
├── docs/                  # Documentation files
├── tests/                 # Test files
└── .venv/                 # Virtual environment
```

## 🆕 New Directory Structure (Latest Update)
The project has been reorganized with a cleaner, more logical directory structure:

```
reports/
├── collections/                    # Telegram message collections
│   ├── raw/                       # Raw collection files from collect command
│   └── *.json                     # Combined collection files
├── dashboards/                     # Dashboard container directory
│   ├── channels/                   # Channel-specific reports (was: reports/channels)
│   │   ├── books/                 # Individual channel directories
│   │   ├── books_magazine/
│   │   ├── Free_Books_life/
│   │   └── SherwinVakiliLibrary/
│   └── html/                       # HTML dashboards (was: reports/dashboards)
│       ├── index.html              # Main dashboard index
│       ├── books_dashboard.html    # Individual channel dashboards
│       └── *.html
├── analysis/                       # Analysis container directory
│   ├── file_messages/              # File-based message analysis
│   └── db_messages/                # Database message analysis
├── channels_overview.json          # Channel overview report
└── logs/                           # Application logs
```

### **🎯 Benefits of New Structure:**
1. **Logical Grouping**: All dashboard-related content is under `reports/dashboards/`
2. **Clear Separation**: Channel reports vs HTML dashboards are clearly separated
3. **Better Organization**: Related files are grouped together
4. **Scalability**: Easy to add new dashboard types in the future

## Configuration-Driven Architecture

### **🎯 Key Principle: No Hardcoded Paths**
All file paths, naming patterns, and directory structures are defined in `config.py` and imported by modules. This ensures:
- **Single Source of Truth**: All file organization rules in one place
- **Easy Maintenance**: Change paths without touching multiple files
- **Environment Flexibility**: Support different directory structures
- **Consistency**: Standardized patterns across all modules

### **📁 Base Path Structure**
```python
# Base paths (config.py)
BASE_DIR = "."
REPORTS_BASE = f"{BASE_DIR}/reports"

# Derived paths built from base
REPORTS_DIR = REPORTS_BASE
COLLECTIONS_DIR = f"{REPORTS_BASE}/collections"
RAW_COLLECTIONS_DIR = f"{COLLECTIONS_DIR}/raw"

# Dashboard structure (new organization)
DASHBOARDS_BASE = f"{REPORTS_BASE}/dashboards"
CHANNELS_DIR = f"{DASHBOARDS_BASE}/channels"           # Channel reports
DASHBOARDS_DIR = f"{DASHBOARDS_BASE}/html"             # HTML dashboards

# Analysis directories (new nested structure)
ANALYSIS_BASE = f"{REPORTS_BASE}/analysis"
FILE_MESSAGES_DIR = f"{ANALYSIS_BASE}/file_messages"    # File-based message analysis
DB_MESSAGES_DIR = f"{ANALYSIS_BASE}/db_messages"        # Database message analysis

TEMPLATES_DIR = f"{BASE_DIR}/templates"
```

### **🔤 File Naming Patterns**
```python
# File naming constants (config.py)
F_PREFIX = "tg"                    # File prefix
F_SEPARATOR = "_"                  # Separator between components
F_EXTENSION = ".json"              # Default file extension

# Dynamic file patterns
COMBINED_FILE_PATTERN = "tg_{channel}_combined.json"
REPORT_FILE_PATTERN = "{channel}_report.json"
SUMMARY_FILE_PATTERN = "{channel}_summary.txt"
DASHBOARD_FILE_PATTERN = "{channel}_dashboard.html"
CHANNELS_OVERVIEW_FILE = "channels_overview.json"
```

### **🔍 File Search Patterns**
```python
# Glob patterns for file discovery (config.py)
RAW_COLLECTION_GLOB = "tg_*.json"
COMBINED_COLLECTION_GLOB = "tg_*_combined.json"
ALL_JSON_GLOB = "*.json"
COMBINED_JSON_GLOB = "*_combined.json"
```

## Core Modules (`modules/`)
All modules now import configuration constants from `config.py`:

### **Data Collection & Processing**
- **`telegram_collector.py`** - Uses `RAW_COLLECTIONS_DIR`, `F_PREFIX`, `F_SEPARATOR`, `F_EXTENSION`
- **`combine_processor.py`** - Uses `RAW_COLLECTIONS_DIR`, `COLLECTIONS_DIR`, `COMBINED_FILE_PATTERN`
- **`import_processor.py`** - Uses `COLLECTIONS_DIR`, `COMBINED_COLLECTION_GLOB`

### **Reporting & Analysis**
- **`channel_reporter.py`** - Uses `CHANNELS_DIR`, `REPORT_FILE_PATTERN`, `SUMMARY_FILE_PATTERN`
- **`report_generator.py`** - Uses `FILE_MESSAGES_DIR`, `COLLECTIONS_DIR`
- **`report_processor.py`** - Uses `COLLECTIONS_DIR`, `CHANNELS_DIR`, `COMBINED_JSON_GLOB`

### **Dashboard & Visualization**
- **`dashboard_generator.py`** - Uses `REPORT_FILE_PATTERN`, `DASHBOARD_FILE_PATTERN`

### **Data Management**
- **`database_service.py`** - Uses API endpoint constants
- **`models.py`** - Data models and schemas
- **`retry_handler.py`** - Retry logic for failed operations

## Templates (`templates/`)
HTML templates for generating interactive dashboards:
- **`base.html`** - Base template with common HTML structure
- **`dashboard_channel.html`** - Individual channel dashboard template
- **`dashboard_index.html`** - Main navigation dashboard template
- **`components/`** - Reusable HTML components
- **`pages/`** - Additional page templates

## Reports Directory (`reports/`)
The `reports/` directory structure is now fully configurable:

### **Collections (`reports/collections/`)**
- **Raw Collections** (`raw/`): Individual message collection files
  - **Pattern**: `tg_{channel}_{min_id}_{max_id}.json`
  - **Example**: `tg_books_1_484.json`
  - **Config**: Uses `RAW_COLLECTIONS_DIR` + `F_PREFIX` + `F_SEPARATOR` + `F_EXTENSION`
  
- **Combined Collections**: Merged datasets from multiple raw files
  - **Pattern**: `tg_{channel}_combined.json`
  - **Example**: `tg_books_combined.json`
  - **Config**: Uses `COLLECTIONS_DIR` + `COMBINED_FILE_PATTERN`

### **Messages (`reports/messages/`)**
Generated message analysis reports:
- **JSON Reports**: `{channel}_report.json` (uses `REPORT_FILE_PATTERN`)
- **Summary Files**: `{channel}_summary.txt` (uses `SUMMARY_FILE_PATTERN`)

### **Channels (`reports/channels/`)**
Directory structure required by dashboard generator:
```
reports/channels/
├── books/
│   └── books_report.json          # Uses REPORT_FILE_PATTERN
├── books_magazine/
│   └── books_magazine_report.json
├── Free_Books_life/
│   └── Free_Books_life_report.json
└── SherwinVakiliLibrary/
    └── SherwinVakiliLibrary_report.json
```

### **Dashboards (`reports/dashboards/`)**
Interactive HTML dashboards:
- **Individual Channel Dashboards**: `{channel}_dashboard.html` (uses `DASHBOARD_FILE_PATTERN`)
- **Main Index**: `index.html`

### **Database Messages (`reports/db_messages/`)**
Exported database content and database-related reports.

### **Logs (`reports/logs/`)**
Application logs and debugging information.

## Data Flow Architecture

### **1. Collection Phase**
```
Telegram API → telegram_collector.py → {RAW_COLLECTIONS_DIR}/
```

### **2. Combination Phase**
```
{RAW_COLLECTIONS_DIR}/ files → combine_processor.py → {COLLECTIONS_DIR}/combined files
```

### **3. Import Phase**
```
{COLLECTIONS_DIR}/combined files → import_processor.py → Database
```

### **4. Reporting Phase**
```
Database → channel_reporter.py → {CHANNELS_DIR}/ (reports/dashboards/channels/)
```

### **5. Dashboard Phase**
```
{CHANNELS_DIR}/ → dashboard_generator.py → {DASHBOARDS_DIR}/ (reports/dashboards/html/)
```

## Configuration Benefits

### **🔄 Easy Path Changes**
```python
# Change all paths at once
BASE_DIR = "/custom/path"  # All paths automatically update
REPORTS_BASE = f"{BASE_DIR}/custom_reports"  # Custom reports location
```

### **🌍 Environment Support**
```python
# Support different environments
BASE_DIR = os.getenv("BASE_PATH", ".")
REPORTS_BASE = os.getenv("REPORTS_PATH", f"{BASE_DIR}/reports")
```

### **📁 Add New Directories**
```python
# Easy to add new subdirectories
NEW_REPORTS_DIR = f"{REPORTS_BASE}/new_category"
CUSTOM_EXPORTS_DIR = f"{REPORTS_BASE}/custom_exports"
```

### **🏷️ Modify File Patterns**
```python
# Change file naming conventions
F_PREFIX = "telegram"  # Change from "tg" to "telegram"
F_SEPARATOR = "-"      # Change from "_" to "-"
```

## Current State Summary

### **✅ Implemented**
- **Configuration-Driven Architecture**: All paths and patterns in `config.py`
- **No Hardcoded Paths**: All modules import from configuration
- **Modular Design**: 10 well-organized core modules
- **Comprehensive Reporting**: Multiple output formats and analysis types
- **Interactive Dashboards**: Plotly charts with professional templates
- **Centralized Configuration**: Single source of truth for file organization

### **🔄 Workflow Status**
- **Collection**: ✅ Fully functional with config-driven paths
- **Combine**: ✅ Fully functional with config-driven paths  
- **Import**: ✅ Fully functional with config-driven paths
- **Report**: ✅ Fully functional with config-driven paths
- **Dashboard**: ✅ Fully functional with config-driven paths

### **📊 Data Volumes**
- **Total Channels**: 4
- **Total Messages**: 2,303
- **Raw Files**: 12 collection files
- **Combined Files**: 4 consolidated datasets
- **Generated Reports**: 8 analysis files
- **Dashboards**: 5 HTML files

## Future Considerations

### **Scalability**
- **Configuration-Driven**: Easy to add new file types and patterns
- **Base Path Flexibility**: Support different directory structures
- **Environment Support**: Easy to support staging, production, etc.

### **Maintenance**
- **Single Source of Truth**: All file organization rules in one place
- **Easy Debugging**: Clear configuration makes path issues obvious
- **Consistent Patterns**: Standardized approach across all modules

### **Extensibility**
- **New File Types**: Easy to add new file patterns and directories
- **Custom Paths**: Support for user-defined directory structures
- **Plugin System**: Configuration can support plugin-based extensions

## Configuration Constants Reference

### **Base Paths**
```python
BASE_DIR = "."                    # Base directory for the project
REPORTS_BASE = "./reports"        # Base reports directory
```

### **Directory Paths**
```python
REPORTS_DIR = "./reports"         # Main reports directory
COLLECTIONS_DIR = "./reports/collections"  # Collections directory
RAW_COLLECTIONS_DIR = "./reports/collections/raw"  # Raw collection files

# Dashboard structure (new organization)
CHANNELS_DIR = "./reports/dashboards/channels"        # Channel reports directory
DASHBOARDS_DIR = "./reports/dashboards/html"          # HTML dashboards directory

# Analysis directories (new nested structure)
FILE_MESSAGES_DIR = "./reports/analysis/file_messages"      # File-based message analysis
DB_MESSAGES_DIR = "./reports/analysis/db_messages"          # Database message analysis
TEMPLATES_DIR = "./templates"     # HTML template files
```

### **File Patterns**
```python
F_PREFIX = "tg"                   # File prefix for Telegram collections
F_SEPARATOR = "_"                 # Separator between filename components
F_EXTENSION = ".json"             # Default file extension
```

### **Dynamic File Patterns**
```python
COMBINED_FILE_PATTERN = "tg_{channel}_combined.json"
REPORT_FILE_PATTERN = "{channel}_report.json"
SUMMARY_FILE_PATTERN = "{channel}_summary.txt"
DASHBOARD_FILE_PATTERN = "{channel}_dashboard.html"
CHANNELS_OVERVIEW_FILE = "channels_overview.json"
```

### **Glob Patterns**
```python
RAW_COLLECTION_GLOB = "tg_*.json"
COMBINED_COLLECTION_GLOB = "tg_*_combined.json"
ALL_JSON_GLOB = "*.json"
COMBINED_JSON_GLOB = "*_combined.json"
```

This configuration-driven approach ensures that the entire file organization system is maintainable, flexible, and consistent across all modules. 🎯✨
