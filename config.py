"""
Configuration file for Telegram Media Messages Tool.
Contains all string constants, default values, and configuration settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = "."
TEMPLATES_DIR = f"{BASE_DIR}/templates"
REPORTS_BASE = f"{BASE_DIR}/reports"

# File paths and directories
REPORTS_DIR = REPORTS_BASE

COLLECTIONS_DIR = f"{REPORTS_BASE}/collections"
RAW_COLLECTIONS_DIR = f"{COLLECTIONS_DIR}/raw"

# Analysis structure (new nested organization)
ANALYSIS_BASE = f"{REPORTS_BASE}/analysis"
FILE_MESSAGES_DIR = f"{ANALYSIS_BASE}/file_messages"    # File-based message analysis
FILES_CHANNELS_DIR = f"{FILE_MESSAGES_DIR}/channels"             # Channel-specific reports
DB_MESSAGES_DIR = f"{ANALYSIS_BASE}/db_messages"        # Database message analysis
DB_CHANNELS_DIR = f"{DB_MESSAGES_DIR}/channels"             # Channel-specific reports

# Analysis file patterns
ANALYSIS_FILE_PATTERN = "{channel}_analysis.json"        # Analysis report files
ANALYSIS_SUMMARY_PATTERN = "{channel}_analysis_summary.txt"  # Analysis summary files

# Analysis glob patterns
ANALYSIS_FILE_GLOB = "*_analysis.json"                   # Analysis file discovery
ANALYSIS_SUMMARY_GLOB = "*_analysis_summary.txt"          # Summary file discovery

# Dashboard structure (nested organization)
DASHBOARDS_BASE = f"{REPORTS_BASE}/dashboards"
DASHBOARDS_DIR = f"{DASHBOARDS_BASE}/html"               # HTML dashboards

# Default export path
DEFAULT_EXPORT_PATH = f"{COLLECTIONS_DIR}/telegram_collection"

# Filename patterns
F_PREFIX = "tg"
F_SEPARATOR = "_"
F_EXTENSION = ".json"

# File naming patterns
COMBINED_FILE_PATTERN = "tg_{channel}_combined.json"
REPORT_FILE_PATTERN = "{channel}_report.json"
SUMMARY_FILE_PATTERN = "{channel}_summary.txt"
DASHBOARD_FILE_PATTERN = "{channel}_dashboard.html"
CHANNELS_OVERVIEW_FILE = "channels_overview.json"

# File glob patterns
RAW_COLLECTION_GLOB = "tg_*.json"
COMBINED_COLLECTION_GLOB = "tg_*_combined.json"
ALL_JSON_GLOB = "*.json"
COMBINED_JSON_GLOB = "*_combined.json"

# Default values
DEFAULT_MAX_MESSAGES = None
DEFAULT_OFFSET_ID = 0
DEFAULT_RATE_LIMIT = 5000  # Increased from 120 to 5000 messages per minute
DEFAULT_SESSION_NAME = "telegram_collector"
DEFAULT_DB_URL = "http://localhost:8000"

# Channel defaults
DEFAULT_CHANNEL = ("@SherwinVakiliLibrary", "@books_magazine", "@books", "@Free_Books_life")
# other channels
OTHER_CHANNELS = ["@books_magazine", "@books","@Free_Books_life" ]
DEFAULT_CHANNEL_PRIORITY = 1

# Rate limiting defaults
DEFAULT_MESSAGES_PER_MINUTE = 5000  # Increased from 1000 to 5000
DEFAULT_DELAY_BETWEEN_CHANNELS = 0.5  # Reduced from 1 to 0.5 seconds
DEFAULT_SESSION_COOLDOWN = 30  # Reduced from 60 to 30 secondssure 

# Google Analytics configuration
DEFAULT_GA_MEASUREMENT_ID = "G-KH0N6NM83F"

# API Endpoints
API_BASE_PATH = "/api/v1/telegram"
API_ENDPOINTS = {
    'messages': f"{API_BASE_PATH}/messages",
    'messages_bulk': f"{API_BASE_PATH}/messages/bulk",
    'stats': f"{API_BASE_PATH}/stats",
    'stats_enhanced': f"{API_BASE_PATH}/stats/enhanced",
    'channels': f"{API_BASE_PATH}/channels",
    'export_all': f"{API_BASE_PATH}/messages/export/all"
}

