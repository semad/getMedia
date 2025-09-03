#!/usr/bin/env python3
"""
Main entry point and CLI for Telegram Media Messages Tool.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_CHANNEL,
    DEFAULT_CHANNEL_PRIORITY,
    DEFAULT_DB_URL,
    DEFAULT_DELAY_BETWEEN_CHANNELS,
    DEFAULT_EXPORT_PATH,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_OFFSET_ID,
    DEFAULT_RATE_LIMIT,
    DEFAULT_SESSION_COOLDOWN,
    DEFAULT_SESSION_NAME,
)
from modules.combine_processor import ( auto_detect_channels_from_raw_advanced,
    combine_existing_collections,
)
from modules.models import ChannelConfig, RateLimitConfig

from modules.telegram_collector import TelegramCollector, export_messages_to_file


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="1.0.0")
@click.help_option("-h", "--help")
def cli():
    """Unified Telegram Media Messages Tool"""
    pass


@cli.command(name="collect")
@click.option("--channels", "-c", help="Comma-separated list of channel usernames")
@click.option(
    "--max-messages", "-m",
    default=DEFAULT_MAX_MESSAGES,
    type=int,
    help="Maximum messages to collect per channel (default: no limit)",
)
@click.option( "--offset-id", "-o", default=DEFAULT_OFFSET_ID, type=int, help="Start collecting from message ID greater than this (default: 0 for complete history)",)
@click.option( "--rate-limit", "-r", default=DEFAULT_RATE_LIMIT, type=int, help="Messages per minute rate limit (default: 120)",)
@click.option( "--session-name", "-s", default=DEFAULT_SESSION_NAME, help="Telegram session name")
@click.option(
    "--file-name",
    "-f",
    default=DEFAULT_EXPORT_PATH,
    help="Export collected messages to JSON file (default: saves to reports/collections/raw/)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.help_option("-h", "--help")
def collect(
    channels, max_messages, offset_id, rate_limit, session_name, file_name, verbose
):
    """Collect messages from Telegram channels."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.info("Verbose logging enabled")
    logger.info("💡 Tip: Press Ctrl+C to gracefully stop collection and save progress")
    # Validate environment variables
    api_id = os.getenv("TG_API_ID")
    api_hash = os.getenv("TG_API_HASH")
    if not api_id or not api_hash:
        logger.error("Missing TG_API_ID or TG_API_HASH environment variables")
        logger.error(
            "Please set these environment variables to use the collect command"
        )
        return
    # Parse and configure channels
    if channels:
        channel_list = [
            ChannelConfig(username=ch.strip()) for ch in channels.split(",")
        ]
    else:
        # Handle both single channel and tuple of channels
        if isinstance(DEFAULT_CHANNEL, tuple):
            channel_list = [
                ChannelConfig(channel, priority=DEFAULT_CHANNEL_PRIORITY)
                for channel in DEFAULT_CHANNEL
            ]
        else:
            channel_list = [
                ChannelConfig(DEFAULT_CHANNEL, priority=DEFAULT_CHANNEL_PRIORITY)
            ]
    # Configure rate limiting
    rate_config = RateLimitConfig(
        messages_per_minute=rate_limit,
        delay_between_channels=DEFAULT_DELAY_BETWEEN_CHANNELS,
        session_cooldown=DEFAULT_SESSION_COOLDOWN,
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
            # Collect messages from each channel
            channel_messages = {}  # Store messages per channel
            for channel in channel_list:
                if not channel.enabled:
                    continue
                logger.info(f"Starting collection from channel: {channel.username}")
                try:
                    messages = await collector.collect_from_channel(
                        channel, max_messages, offset_id
                    )
                    channel_messages[channel.username] = messages
                    all_messages.extend(messages)
                    logger.info(
                        f"Collected {len(messages)} messages from {channel.username}"
                    )
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
    all_messages = []
    channel_messages = {}
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
                # Export to raw directory for new collections
                raw_export_path = Path("reports/collections/raw")
                raw_export_path.mkdir(parents=True, exist_ok=True)
                export_filename = export_messages_to_file(
                    messages, str(raw_export_path), single_channel_list
                )
                if export_filename:
                    exported_files.append(export_filename)
                    logger.info(
                        f"📁 Messages from {channel_username} exported to raw directory: {export_filename}"
                    )
    # Also export combined file if custom filename is provided
    if all_messages and file_name != DEFAULT_EXPORT_PATH:
        export_filename = export_messages_to_file(all_messages, file_name, channel_list)
        if export_filename:
            exported_files.append(export_filename)
            logger.info(f"📁 Combined messages exported to: {export_filename}")
    # Print summary
    logger.info("🎯 Collection completed!")
    logger.info(f"📊 Total messages collected: {len(all_messages)}")
    enabled_channels = [
        ch for ch in channel_list if hasattr(ch, "enabled") and ch.enabled
    ]
    logger.info(f"📺 Channels processed: {len(enabled_channels)}")
    if exported_files:
        logger.info(f"📁 Files created: {len(exported_files)}")
        for file_path in exported_files:
            logger.info(f"   📄 {file_path}")


@cli.command(name="combine")
@click.option(
    "--channels",
    "-c",
    help="Comma-separated list of channel usernames (auto-detects if not specified)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.help_option("-h", "--help")
def combine_collections(channels, verbose):
    """Combine existing collection files for the same channel into consolidated files."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.info("Verbose logging enabled")
    logger.info("🔄 Combine mode enabled - processing collection files...")
    # Handle combine option - combine existing collection files for the same channel
    if channels:
        logger.info("🔄 Using specified channels for combination...")
        channel_list = [ch.strip() for ch in channels.split(",")]
        # Use the combine function
        result = combine_existing_collections(channels=channel_list, verbose=verbose)
        combined_files = result.get("combined_files", [])
        if combined_files:
            logger.info(f"✅ Combined {len(combined_files)} collection files")
            for combined_file in combined_files:
                logger.info(f"📁 Combined file created: {combined_file}")
        else:
            logger.warning("⚠️  No existing collection files found to combine")
    else:
        logger.info("🔄 Auto-detecting channels from raw directory...")
        auto_detected_channels = auto_detect_channels_from_raw_advanced(
            Path("reports/collections/raw")
        )
        if auto_detected_channels:
            logger.info(
                f"🔍 Auto-detected {len(auto_detected_channels)} channels: {', '.join(auto_detected_channels)}"
            )
            # Use the combine function
            result = combine_existing_collections(
                channels=auto_detected_channels, verbose=verbose
            )
            combined_files = result.get("combined_files", [])
            if combined_files:
                logger.info(f"✅ Combined {len(combined_files)} collection files")
                for combined_file in combined_files:
                    logger.info(f"📁 Combined file created: {combined_file}")
            else:
                logger.warning("⚠️  No existing collection files found to combine")
        else:
            logger.warning("⚠️  No channels detected in raw directory")
    logger.info("🎯 Combine operation completed!")


@cli.command(name="analysis")
@click.option("--channels", "-c", 
              help="Comma-separated list of channel usernames to analyze")
@click.option("--analysis-types", "-t",
              help="Comma-separated list of analysis types: filename,filesize,message (default: all)")
@click.option("--output-dir", "-o", 
              type=click.Path(), 
              help="Output directory for analysis results (default: analysis_output_<timestamp>)")
@click.option("--chunk-size", 
              type=int, 
              default=10000,
              help="Chunk size for processing large datasets (default: 10000)")
@click.option("--verbose", "-v", 
              is_flag=True, 
              help="Enable verbose logging output")
@click.help_option("-h", "--help")
def analysis(channels, analysis_types, output_dir, chunk_size, verbose):
    """Run comprehensive analysis on Telegram channel data.
    
    This command analyzes collected Telegram channel data for filename patterns,
    filesize distributions, and message content patterns.
    
    Examples:
        python main.py analysis                           # Analyze all data with default settings
        python main.py analysis --channels "@channel1,@channel2"    # Analyze specific channels
        python main.py analysis --analysis-types filename,filesize  # Run specific analysis types
        python main.py analysis --output-dir ./results    # Specify output directory
        python main.py analysis --chunk-size 5000         # Use smaller chunks for large datasets
        python main.py analysis --verbose                 # Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    try:
        # Import analysis modules
        from modules.analysis import create_analysis_config, AnalysisOrchestrator
        
        # Create configuration
        config_kwargs = {
            "verbose": verbose,
            "chunk_size": chunk_size
        }
        
        # Add output directory if provided
        if output_dir:
            config_kwargs["output_dir"] = output_dir
        
        # Create configuration
        config = create_analysis_config(**config_kwargs)
        logger.info(f"Analysis configuration: {config}")
        
        # Create orchestrator
        orchestrator = AnalysisOrchestrator(config)
        
        # Prepare analysis parameters
        analysis_kwargs = {}
        
        if channels:
            channel_list = [ch.strip() for ch in channels.split(",")]
            analysis_kwargs["channels"] = channel_list
        
        # Set analysis types
        if analysis_types:
            analysis_types_list = [t.strip() for t in analysis_types.split(",")]
            # Validate analysis types
            valid_types = ["filename", "filesize", "message"]
            invalid_types = [t for t in analysis_types_list if t not in valid_types]
            if invalid_types:
                logger.error(f"❌ Invalid analysis types: {invalid_types}. Valid types: {valid_types}")
                return
            analysis_kwargs["analysis_types"] = analysis_types_list
        else:
            # Default to all analysis types
            analysis_kwargs["analysis_types"] = ["filename", "filesize", "message"]
        
        # Run analysis
        logger.info("🚀 Starting comprehensive analysis...")
        results = asyncio.run(orchestrator.run_comprehensive_analysis(**analysis_kwargs))
        
        if "error" in results:
            logger.error(f"❌ Analysis failed: {results['error']}")
            return
        
        # Display summary
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📊 Analysis ID: {results.get('analysis_id', 'N/A')}")
        logger.info(f"📈 Data sources found: {len(results.get('data_sources', []))}")
        logger.info(f"⏱️  Processing time: {results.get('performance_stats', {}).get('total_time_seconds', 'N/A')} seconds")
        
        # Show analysis results
        analysis_results = results.get('analysis_results', {})
        if analysis_results:
            logger.info("🔍 Analysis results:")
            for analysis_type, result in analysis_results.items():
                if hasattr(result, 'total_files') or hasattr(result, 'total_messages'):
                    count = getattr(result, 'total_files', getattr(result, 'total_messages', 0))
                    logger.info(f"   📊 {analysis_type}: {count} items analyzed")
        
        # Show output paths
        output_paths = results.get('output_paths', {})
        if output_paths:
            logger.info("📁 Output files created:")
            for report_type, path in output_paths.items():
                logger.info(f"   📄 {report_type}: {path}")
        
    except Exception as e:
        logger.error(f"❌ Analysis command failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


@cli.command(name="import")
@click.argument("import_file", type=click.Path(exists=True), required=False)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.help_option("-h", "--help")
def import_file(import_file, verbose):
    """Import messages from existing JSON file to database.

    If no file is specified, imports from all available combined collections.
    Examples:
        python main.py import                           # Import from all combined collections
        python main.py import file.json                 # Import from specific file
        python main.py import --verbose                 # Import all with verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.info("Verbose logging enabled")
    # Get database URL from environment
    db_url = os.getenv("TELEGRAM_DB_URL", DEFAULT_DB_URL)
    logger.info(f"Connecting to database at: {db_url}")

    try:
        if import_file:
            # Import from specific file
            logger.info(f"🔄 Importing messages from specified file: {import_file}")
            from modules.import_processor import import_single_file

            import_result = import_single_file(import_file, db_url, logger)
            if import_result:
                logger.info(
                    f"✅ Successfully imported messages from {import_file} to database"
                )
            else:
                logger.error(
                    f"❌ Failed to import messages from {import_file} to database"
                )
        else:
            # Import from all available combined collections
            logger.info(
                "🔄 No file specified, importing from all available combined collections..."
            )
            from modules.import_processor import import_all_combined_collections

            import_result = import_all_combined_collections(db_url, logger, verbose)
            if import_result:
                logger.info(
                    "✅ Successfully imported all combined collections to database"
                )
            else:
                logger.error("❌ Failed to import some or all combined collections")
    except Exception as e:
        logger.error(f"Failed to import messages: {e}")
        if verbose:
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    cli()
