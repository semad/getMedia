#!/usr/bin/env python3
"""
Main entry point and CLI for Telegram Media Messages Tool.
"""

import sys
import os
import click
import asyncio
import logging
import json
import signal
import re
from datetime import datetime, timezone
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from modules.import_processor import run_import
from modules.telegram_collector import TelegramCollector, DatabaseChecker, convert_messages_to_dataframe_format, export_messages_to_file
from modules.models import ChannelConfig, RateLimitConfig
from modules.database_service import TelegramDBService
from modules.telegram_analyzer import TelegramDataAnalyzer


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.version_option(version='1.0.0')
@click.help_option('-h', '--help')
def cli():
    """Unified Telegram Media Messages Tool"""
    pass


@cli.command(name='collect')
@click.option('--channels', '-c', 
              help='Comma-separated list of channel usernames')
@click.option('--max-messages', '-m', default=DEFAULT_MAX_MESSAGES, type=int,
              help='Maximum messages to collect per channel (default: no limit)')
@click.option('--offset-id', '-o', default=DEFAULT_OFFSET_ID, type=int,
              help='Start collecting from message ID greater than this (default: 0 for complete history)')
@click.option('--rate-limit', '-r', default=DEFAULT_RATE_LIMIT, type=int,
              help='Messages per minute rate limit (default: 120)')
@click.option('--session-name', '-s', default=DEFAULT_SESSION_NAME,
              help='Telegram session name')
@click.option('--file-name', '-f', default=DEFAULT_EXPORT_PATH, 
              help='Export collected messages to JSON file')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging output')
@click.help_option('-h', '--help')
def collect(channels, max_messages, offset_id, rate_limit, session_name, file_name, verbose):
    """Collect messages from Telegram channels."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    logger.info("💡 Tip: Press Ctrl+C to gracefully stop collection and save progress")
    
    # Validate environment variables
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    db_url = os.getenv('TELEGRAM_DB_URL', DEFAULT_DB_URL)
    
    if not api_id or not api_hash:
        logger.error("Missing TG_API_ID or TG_API_HASH environment variables")
        logger.error("Please set these environment variables to use the collect command")
        return
    
    # Parse and configure channels
    if channels:
        channel_list = [ChannelConfig(username=ch.strip()) for ch in channels.split(',')]
    else:
        channel_list = [ChannelConfig(DEFAULT_CHANNEL, priority=DEFAULT_CHANNEL_PRIORITY)]
    
    # Configure rate limiting
    rate_config = RateLimitConfig(
        messages_per_minute=rate_limit if rate_limit != DEFAULT_RATE_LIMIT else DEFAULT_MESSAGES_PER_MINUTE,
        delay_between_channels=DEFAULT_DELAY_BETWEEN_CHANNELS,
        session_cooldown=DEFAULT_SESSION_COOLDOWN
    )
    
    async def collect_messages():
        """Main collection logic."""
        collector = TelegramCollector(rate_config)
        all_messages = []
        try:
            # Initialize collector
            if not await collector.initialize(api_id, api_hash, session_name):
                logger.error("Failed to initialize Telegram collector")
                return all_messages
            
            # Use provided offset ID directly
            actual_offset_id = offset_id
            
            # Collect messages from each channel
            channel_messages = {}  # Store messages per channel
            for channel in channel_list:
                if not channel.enabled:
                    continue
                    
                logger.info(f"Starting collection from channel: {channel.username}")
                try:
                    messages = await collector.collect_from_channel(channel, max_messages, actual_offset_id)
                    channel_messages[channel.username] = messages
                    all_messages.extend(messages)
                    logger.info(f"Collected {len(messages)} messages from {channel.username}")
                except asyncio.CancelledError:
                    logger.info("🛑 Collection cancelled, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error collecting from {channel.username}: {e}")
                    channel_messages[channel.username] = []
            
            # Collection completed
            if all_messages:
                logger.info(f"✅ Collected {len(all_messages)} messages successfully")
                
        except Exception as e:
            logger.error(f"Error during collection: {e}")
        finally:
            await collector.close()
            
        return all_messages, channel_messages
    

    

    # Run collection
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(collect_messages())
        if isinstance(result, tuple):
            all_messages, channel_messages = result
        else:
            all_messages = result
            channel_messages = {}
    except KeyboardInterrupt:
        logger.info("🛑 Collection interrupted by user")
        all_messages = []
        channel_messages = {}
    finally:
        if loop:
            loop.close()
    
    # Export results - create separate files for each channel
    exported_files = []
    if channel_messages:
        for channel_username, messages in channel_messages.items():
            if messages:  # Only export if there are messages
                # Create single-channel list for this export
                single_channel_list = [ChannelConfig(channel_username)]
                export_filename = export_messages_to_file(messages, DEFAULT_EXPORT_PATH, single_channel_list, logger)
                if export_filename:
                    exported_files.append(export_filename)
                    logger.info(f"📁 Messages from {channel_username} exported to: {export_filename}")
    
    # Also export combined file if custom filename is provided
    if all_messages and file_name != DEFAULT_EXPORT_PATH:
        export_filename = export_messages_to_file(all_messages, file_name, channel_list, logger)
        if export_filename:
            exported_files.append(export_filename)
            logger.info(f"📁 Combined messages exported to: {export_filename}")
    
    # Print summary
    logger.info("Collection completed!")
    logger.info(f"Total messages collected: {len(all_messages)}")
    logger.info(f"Channels processed: {len([ch for ch in channel_list if ch.enabled])}")
    if exported_files:
        logger.info(f"Files created: {len(exported_files)}")


@cli.command(name='import')
@click.argument('import_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.help_option('-h', '--help')
def import_file(import_file, verbose):
    """Import messages from existing JSON file to database."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    # Get database URL from environment
    db_url = os.getenv('TELEGRAM_DB_URL', DEFAULT_DB_URL)
    logger.info(f"Connecting to database at: {db_url}")
    
    try:
        logger.info(f"🔄 Importing messages from existing file: {import_file}")
        
        # Load and validate the JSON file
        logger.info("🔍 Loading and validating JSON file...")
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate that this is in the expected structured DataFrame format
            if not (data.get('metadata', {}).get('data_format') == 'structured_dataframe' and 
                    data.get('messages') and 
                    isinstance(data['messages'], list)):
                logger.error("❌ File is not in the expected structured DataFrame format")
                logger.error("Expected: metadata.data_format = 'structured_dataframe' and messages array")
                return
            
            logger.info(f"✅ JSON file validated - {len(data['messages'])} messages in structured format")
            import_file_to_use = import_file
                
        except Exception as e:
            logger.error(f"Failed to read or validate JSON file: {e}")
            return
        
        # Run the import synchronously since we're not in an async context here
        from modules.import_processor import run_import
        import_result = asyncio.run(run_import(import_file_to_use, db_url))
        
        if import_result:
            logger.info(f"✅ Successfully imported messages from {import_file} to database")
        else:
            logger.error(f"❌ Failed to import messages from {import_file} to database")
        
        # Clean up temporary file if it was created
        # The import_file_to_use is now the original file, so no cleanup needed here
                
    except Exception as e:
        logger.error(f"Failed to import messages from {import_file} to database: {e}")


@cli.command(name='analyze')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--dashboard', '-d', is_flag=True, help='Generate interactive HTML dashboard')
@click.option('--output', '-o', default=DEFAULT_ANALYSIS_OUTPUT,
              help='Output HTML file path for dashboard (default: ./reports/analysis/telegram_analysis_dashboard.html)')
@click.option('--file', '-f', type=click.Path(exists=True), help='Analyze data from file instead of database')
@click.option('--summary', '-s', is_flag=True, help='Generate summary report')
@click.help_option('-h', '--help')
def analyze(verbose, dashboard, output, file, summary):
    """Analyze Telegram message data from database or files with comprehensive reports.
    
    This unified command handles:
    - Database analysis (default): Show database statistics and summary information
    - File analysis: Analyze data from JSON/CSV files with detailed reports
    - Interactive dashboard generation (planned feature)
    
    Use --file to analyze data from files instead of database.
    Use --summary to generate concise summary reports.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    # Ensure analysis directory exists
    analysis_dir = os.path.dirname(output)
    if analysis_dir and not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir, exist_ok=True)
        logger.info(f"Created analysis directory: {analysis_dir}")
    
    if file:
        # File analysis mode
        logger.info(f"FILE ANALYSIS MODE - Analyzing data from: {file}")
        from modules.analyze_processor import analyze_file
        analyze_file(file, logger, dashboard, output, summary)
    else:
        # Default: Database analysis mode
        logger.info("DATABASE ANALYSIS MODE - Analyzing database statistics")
        from modules.analyze_processor import analyze_database
        analyze_database(logger, summary)


if __name__ == '__main__':
    cli()
