#!/usr/bin/env python3
"""
Analyze text and caption field sizes in Telegram export data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_text_sizes(file_path: str):
    """Analyze text and caption field sizes in the export data."""
    
    logger.info(f"Analyzing text sizes in: {file_path}")
    
    # Load the data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' in data:
        messages = data['data']
        logger.info(f"Found {len(messages)} messages in data array")
    elif 'messages' in data:
        messages = data['messages']
        logger.info(f"Found {len(messages)} messages in messages array")
    else:
        logger.error("No data or messages array found in file")
        return
    
    # Analyze text and caption fields
    text_lengths = []
    caption_lengths = []
    missing_text = 0
    missing_caption = 0
    very_long_text = 0
    very_long_caption = 0
    
    for i, msg in enumerate(messages):
        # Analyze text field
        if 'text' in msg and msg['text']:
            text_len = len(str(msg['text']))
            text_lengths.append(text_len)
            if text_len > 10000:  # Very long text
                very_long_text += 1
                logger.warning(f"Message {i}: Very long text ({text_len} chars): {str(msg['text'])[:100]}...")
        else:
            missing_text += 1
        
        # Analyze caption field
        if 'caption' in msg and msg['caption']:
            caption_len = len(str(msg['caption']))
            caption_lengths.append(caption_len)
            if caption_len > 5000:  # Very long caption
                very_long_caption += 1
                logger.warning(f"Message {i}: Very long caption ({caption_len} chars): {str(msg['caption'])[:100]}...")
        else:
            missing_caption += 1
    
    # Calculate statistics
    if text_lengths:
        logger.info(f"\n=== TEXT FIELD ANALYSIS ===")
        logger.info(f"Total messages with text: {len(text_lengths)}")
        logger.info(f"Messages missing text: {missing_text}")
        logger.info(f"Average text length: {sum(text_lengths) / len(text_lengths):.1f} characters")
        logger.info(f"Min text length: {min(text_lengths)} characters")
        logger.info(f"Max text length: {max(text_lengths)} characters")
        logger.info(f"Very long texts (>10k chars): {very_long_text}")
        
        # Check for extremely long texts
        if max(text_lengths) > 100000:
            logger.warning(f"WARNING: Found extremely long text ({max(text_lengths)} characters)")
    
    if caption_lengths:
        logger.info(f"\n=== CAPTION FIELD ANALYSIS ===")
        logger.info(f"Total messages with caption: {len(caption_lengths)}")
        logger.info(f"Messages missing caption: {missing_caption}")
        logger.info(f"Average caption length: {sum(caption_lengths) / len(caption_lengths):.1f} characters")
        logger.info(f"Min caption length: {min(caption_lengths)} characters")
        logger.info(f"Max caption length: {max(caption_lengths)} characters")
        logger.info(f"Very long captions (>5k chars): {very_long_caption}")
        
        # Check for extremely long captions
        if max(caption_lengths) > 50000:
            logger.warning(f"WARNING: Found extremely long caption ({max(caption_lengths)} characters)")
    
    # Check database constraints
    logger.info(f"\n=== DATABASE CONSTRAINT CHECK ===")
    logger.info(f"PostgreSQL TEXT field: Unlimited length")
    logger.info(f"PostgreSQL VARCHAR field: Limited to specified length")
    
    # Check for potential issues
    issues = []
    if very_long_text > 0:
        issues.append(f"Found {very_long_text} very long text fields (>10k chars)")
    if very_long_caption > 0:
        issues.append(f"Found {very_long_caption} very long caption fields (>5k chars)")
    if missing_text > 0:
        issues.append(f"Found {missing_text} messages without text")
    
    if issues:
        logger.warning(f"\nPOTENTIAL ISSUES:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info(f"\nNo obvious text/caption size issues found.")

if __name__ == "__main__":
    analyze_text_sizes("reports/exports/current_db_export.json")
