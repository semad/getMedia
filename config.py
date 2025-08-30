"""
Configuration file for Telegram Media Messages Tool.
Contains all string constants, default values, and configuration settings.
"""

import os
from pathlib import Path

# File paths and directories
REPORTS_DIR = "./reports"
COLLECTIONS_DIR = "./reports/collections"
ANALYSIS_DIR = "./reports/analysis"
DEFAULT_EXPORT_PATH = "./reports/collections/telegram_collection"
DEFAULT_ANALYSIS_OUTPUT = "./reports/analysis/telegram_analysis_dashboard.html"

# Filename patterns
F_PREFIX = "tg"
F_SEPARATOR = "_"
F_EXTENSION = ".json"

# Default values
DEFAULT_MAX_MESSAGES = None
DEFAULT_OFFSET_ID = 0
DEFAULT_RATE_LIMIT = 120
DEFAULT_SESSION_NAME = "telegram_collector"
DEFAULT_DB_URL = "http://localhost:80"

# Channel defaults
DEFAULT_CHANNEL = "@SherwinVakiliLibrary"
DEFAULT_CHANNEL_PRIORITY = 1

# Rate limiting defaults
DEFAULT_MESSAGES_PER_MINUTE = 1000
DEFAULT_DELAY_BETWEEN_CHANNELS = 1
DEFAULT_SESSION_COOLDOWN = 60








