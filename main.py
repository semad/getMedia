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
              default="all",
              help="Comma-separated channel list or 'all'")
@click.option("--verbose", "-v", 
              is_flag=True, 
              help="Enable verbose logging")
@click.option("--api", 
              is_flag=True, 
              help="Use API source only, no file source")
@click.option("--file", 
              is_flag=True, 
              help="Use file source only, no API source")
@click.help_option("-h", "--help")
def analysis(channels, verbose, api, file):
    """Run comprehensive analysis on Telegram channel data.
    
    This command analyzes collected Telegram channel data for filename patterns,
    filesize distributions, and message content patterns.
    
    Examples:
        python main.py analysis                           # Default behavior (file source only)
        python main.py analysis --channels @SherwinVakiliLibrary  # Analysis with specific channels
        python main.py analysis --verbose                 # Analysis with verbose logging
        python main.py analysis --channels @SherwinVakiliLibrary --verbose  # Analysis with specific channels and verbose logging
        python main.py analysis --file                   # Analysis using file source only 
        python main.py analysis --channels @SherwinVakiliLibrary --file  # Analysis with specific channels using file source only
        python main.py analysis --api                    # Analysis using API source only
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    try:
        # Import analysis modules
        from modules.analysis_processor import create_analysis_config, run_advanced_intermediate_analysis
        
        # Create configuration
        config_kwargs = {
            "verbose": verbose,
            "enable_file_source": not api,  # Default to file source unless --api is specified
            "enable_api_source": api
        }
        
        # Add channels if specified
        if channels and channels != "all":
            channel_list = [ch.strip() for ch in channels.split(",")]
            config_kwargs["channels"] = channel_list
        
        # Create configuration
        config = create_analysis_config(**config_kwargs)
        logger.info(f"Analysis configuration: {config}")
        
        # Run analysis
        logger.info("🚀 Starting comprehensive analysis...")
        results = asyncio.run(run_advanced_intermediate_analysis(config))
        
        if "error" in results:
            logger.error(f"❌ Analysis failed: {results['error']}")
            return
        
        # Display summary
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📊 Channels processed: {len(results)}")
        
        # Show analysis results
        for channel_name, result in results.items():
            if "error" in result:
                logger.error(f"❌ Error processing {channel_name}: {result['error']}")
                continue
            
            logger.info(f"📈 {channel_name}:")
            logger.info(f"   📄 Files analyzed: {result.get('metadata', {}).get('total_records', 'N/A')}")
            
            # Show output paths
            output_files = result.get('output_files', {})
            if output_files:
                logger.info(f"   📁 Output files:")
                for report_type, path in output_files.items():
                    logger.info(f"      📄 {report_type}: {path}")
        
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


@cli.command()
@click.option('--input-dir', '-i', 'input_dir',
              default=None, help='Directory containing analysis results')
@click.option('--output-dir', '-o', 'output_dir',
              default=None, help='Directory to save generated HTML files')
@click.option('--channels', '-c', 'channels',
              default=None, help='Comma-separated list of channels to process')
@click.option('--verbose', '-v', 'verbose',
              is_flag=True, help='Enable verbose logging output')
def dashboard(input_dir, output_dir, channels, verbose):
    """Generate HTML dashboard from analysis results."""
    # Setup logging if setup_logging function exists, otherwise use basic logging
    try:
        setup_logging(verbose)
    except ImportError:
        import logging
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    
    try:
        from modules.dashboard_processor import DashboardProcessor
        
        processor = DashboardProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            channels=channels,
            verbose=verbose
        )
        
        processor.process()
        click.echo("Dashboard generated successfully!")
        
    except ImportError as e:
        click.echo(f"Error importing dashboard modules: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except FileNotFoundError as e:
        click.echo(f"Required file or directory not found: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except PermissionError as e:
        click.echo(f"Permission denied: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
    except Exception as e:
        click.echo(f"Error generating dashboard: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    cli()
