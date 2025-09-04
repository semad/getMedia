#!/usr/bin/env python3
"""
Configuration validation script for dashboard implementation.
Ensures all required constants are properly defined in config.py.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_config():
    """Validate that all dashboard configuration constants are properly defined."""
    try:
        from config import (
            DASHBOARD_INPUT_DIR, DASHBOARD_OUTPUT_DIR,
            DASHBOARD_INDEX_FILENAME, DASHBOARD_CSS_FILENAME,
            DASHBOARD_JS_FILENAME, DASHBOARD_DATA_FILENAME,
            DASHBOARD_CSS_PATH, DASHBOARD_JS_PATH,
            DASHBOARD_GA_MEASUREMENT_ID, DASHBOARD_GA_ENABLED,
            DASHBOARD_SUPPORTED_ANALYSIS_TYPES, DASHBOARD_SUPPORTED_SOURCE_TYPES,
            DASHBOARD_MAX_CHANNEL_NAME_LENGTH, DASHBOARD_CHARTJS_CDN_URL,
            TEMPLATES_DIR, ANALYSIS_BASE, DASHBOARDS_DIR
        )
        
        print("✅ All dashboard configuration constants are properly defined!")
        print(f"   - Input directory: {DASHBOARD_INPUT_DIR}")
        print(f"   - Output directory: {DASHBOARD_OUTPUT_DIR}")
        print(f"   - Templates directory: {TEMPLATES_DIR}")
        print(f"   - Analysis base: {ANALYSIS_BASE}")
        print(f"   - Dashboards directory: {DASHBOARDS_DIR}")
        print(f"   - Supported analysis types: {len(DASHBOARD_SUPPORTED_ANALYSIS_TYPES)}")
        print(f"   - Supported source types: {len(DASHBOARD_SUPPORTED_SOURCE_TYPES)}")
        print(f"   - Google Analytics enabled: {DASHBOARD_GA_ENABLED}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration validation failed: {e}")
        print("Please ensure all dashboard constants are added to config.py")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        return False

if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
