# Project Directory Tree

## Current Project Structure (as of September 1, 2025)

```
getMedia/
├── 📁 .git/                          # Git repository
├── 📁 .vscode/                       # VS Code configuration
├── 📁 .venv/                         # Python virtual environment
├── 📁 __pycache__/                   # Python bytecode cache
├── 📁 .mypy_cache/                   # Type checking cache
├── 📁 .ruff_cache/                    # Ruff linting cache
├── 📁 .pytest_cache/                  # Pytest cache
├── 📁 tests/                          # Test files
├── 📁 Archive/                        # Archived/legacy files
├── 📁 FastAPI/                        # FastAPI backend
├── 📁 sortMedia/                      # Media sorting utilities
├── 📁 templates/                      # HTML templates for dashboards
│   ├── 📄 base.html                   # Base template
│   ├── 📄 dashboard_channel.html      # Channel dashboard template
│   ├── 📄 dashboard_index.html        # Main index template
│   ├── 📁 components/                 # Reusable HTML components
│   └── 📁 pages/                      # Additional page templates
├── 📁 modules/                        # Core Python modules
│   ├── 📄 __init__.py                 # Module initialization
│   ├── 📄 telegram_collector.py       # Telegram API integration
│   ├── 📄 combine_processor.py        # Data combination logic
│   ├── 📄 import_processor.py         # Database import logic
│   ├── 📄 channel_reporter.py         # Report generation
│   ├── 📄 report_generator.py         # Multi-format reports
│   ├── 📄 report_processor.py         # Report processing
│   ├── 📄 dashboard_generator.py      # Dashboard creation
│   ├── 📄 database_service.py         # Database operations
│   ├── 📄 models.py                   # Data models
│   └── 📄 retry_handler.py            # Retry logic
├── 📁 docs/                           # Documentation
│   ├── 📄 manage_media_messages.md    # Media management guide
│   ├── 📄 README_MODULAR.md           # Modular architecture guide
│   ├── 📄 TODO.md                     # Development tasks
│   ├── 📄 GOOGLE_ANALYTICS_SETUP.md  # Analytics setup
│   ├── 📄 FILE_ORGANIZATION_DESIGN.md # File organization guide
│   └── 📄 DIRECTORY_TREE.md           # This file
├── 📁 reports/                        # All generated outputs
│   ├── 📄 README.md                   # Reports documentation
│   ├── 📄 channels_overview.json      # Channel summary
│   ├── 📁 collections/                # Telegram message collections
│   │   ├── 📄 README.md               # Collections documentation
│   │   ├── 📄 tg_books_combined.json  # Combined books data
│   │   ├── 📄 tg_books_magazine_combined.json
│   │   ├── 📄 tg_Free_Books_life_combined.json
│   │   ├── 📄 tg_SherwinVakiliLibrary_combined.json
│   │   └── 📁 raw/                    # Raw collection files
│   │       ├── 📄 tg_books_1_484.json
│   │       ├── 📄 tg_books_312_484.json
│   │       ├── 📄 tg_books_450_483.json
│   │       ├── 📄 tg_books_magazine_3195_4222.json
│   │       ├── 📄 tg_books_magazine_4172_4222.json
│   │       ├── 📄 tg_books_magazine_4207_4216.json
│   │       ├── 📄 tg_Free_Books_life_1_118.json
│   │       ├── 📄 tg_Free_Books_life_67_118.json
│   │       ├── 📄 tg_Free_Books_life_109_118.json
│   │       ├── 📄 tg_SherwinVakiliLibrary_149264_150300.json
│   │       ├── 📄 tg_SherwinVakiliLibrary_150251_150300.json
│   │       └── 📄 tg_SherwinVakiliLibrary_150291_150300.json
│   ├── 📁 channels/                   # Dashboard-compatible structure
│   │   ├── 📁 books/
│   │   │   └── 📄 books_report.json
│   │   ├── 📁 books_magazine/
│   │   │   └── 📄 books_magazine_report.json
│   │   ├── 📁 Free_Books_life/
│   │   │   └── 📄 Free_Books_life_report.json
│   │   └── 📁 SherwinVakiliLibrary/
│   │       └── 📄 SherwinVakiliLibrary_report.json
│   ├── 📁 messages/                   # Generated analysis reports
│   │   ├── 📄 tg_books_report.json
│   │   ├── 📄 tg_books_summary.txt
│   │   ├── 📄 tg_books_magazine_report.json
│   │   ├── 📄 tg_books_magazine_summary.txt
│   │   ├── 📄 tg_Free_Books_life_report.json
│   │   ├── 📄 tg_Free_Books_life_summary.txt
│   │   ├── 📄 tg_SherwinVakiliLibrary_report.json
│   │   └── 📄 tg_SherwinVakiliLibrary_summary.txt
│   ├── 📁 dashboards/                 # Interactive HTML dashboards
│   │   ├── 📄 index.html              # Main navigation dashboard
│   │   ├── 📄 books_dashboard.html    # Books channel dashboard
│   │   ├── 📄 books_magazine_dashboard.html
│   │   ├── 📄 Free_Books_life_dashboard.html
│   │   └── 📄 SherwinVakiliLibrary_dashboard.html
│   ├── 📁 db_messages/                # Database exports
│   └── 📁 logs/                       # Application logs
├── 📄 main.py                         # Main CLI application (24KB, 613 lines)
├── 📄 config.py                        # Configuration constants (1.7KB, 53 lines)
├── 📄 requirements.txt                 # Python dependencies
├── 📄 pyproject.toml                  # Project configuration
├── 📄 README.md                        # Project documentation (11KB, 461 lines)
├── 📄 .gitignore                       # Git ignore patterns
├── 📄 .python-version                  # Python version specification
├── 📄 telegram_collector.session       # Telegram session file (192KB)
├── 📄 telegram_collector.log           # Collection logs (13MB)
├── 📄 .coverage                        # Test coverage data
├── 📄 run_tests.py                     # Test runner
├── 📄 convert_legacy_json.py           # Legacy data converter
├── 📄 analyze_text_sizes.py            # Text analysis utility
├── 📄 pandas_json_demo.py              # Pandas JSON demo
├── 📄 test_pandas_warnings.py          # Pandas warning tests
├── 📄 activate_env.sh                  # Environment activation script
└── 📄 getMedia.code-workspace          # VS Code workspace file
```

## Key Statistics

### 📊 File Counts
- **Total Files**: 50+ files
- **Python Modules**: 10 core modules
- **HTML Templates**: 5 template files
- **Generated Reports**: 8 analysis files
- **Dashboards**: 5 HTML files
- **Raw Collections**: 12 collection files
- **Combined Collections**: 4 consolidated files

### 📁 Directory Structure
- **Root Level**: 8 main directories
- **Reports Subdirectories**: 6 specialized directories
- **Module Organization**: 4 logical groupings
- **Template Structure**: 3 template categories

### 💾 Data Volumes
- **Total Messages**: 2,303
- **Channels**: 4 active channels
- **Collection Files**: 12 raw + 4 combined
- **Report Files**: 8 analysis files
- **Dashboard Files**: 5 interactive HTML files

## File Type Distribution

### 📄 Python Files (.py)
- **Main Application**: 1 file
- **Core Modules**: 10 files
- **Utility Scripts**: 4 files
- **Test Files**: Multiple test files

### 📄 JSON Files (.json)
- **Raw Collections**: 12 files
- **Combined Collections**: 4 files
- **Analysis Reports**: 4 files
- **Configuration**: 1 file

### 📄 HTML Files (.html)
- **Dashboard Templates**: 2 files
- **Generated Dashboards**: 5 files
- **Base Template**: 1 file

### 📄 Markdown Files (.md)
- **Documentation**: 6 files
- **README**: 1 file

### 📄 Text Files (.txt)
- **Summary Reports**: 4 files
- **Logs**: Multiple log files

## Naming Conventions

### 🔤 File Prefixes
- **Telegram Collections**: `tg_` prefix
- **Reports**: `tg_{channel}_report.json`
- **Summaries**: `tg_{channel}_summary.txt`
- **Combined**: `tg_{channel}_combined.json`

### 🔗 Separators
- **Underscore**: `_` for component separation
- **Hyphen**: `-` for dashboard files
- **Slash**: `/` for directory paths

### 📁 Directory Patterns
- **Reports**: `reports/{category}/`
- **Modules**: `modules/{functionality}.py`
- **Templates**: `templates/{type}.html`
- **Channels**: `reports/channels/{channel_name}/`
