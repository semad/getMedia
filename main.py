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
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_MAX_MESSAGES, DEFAULT_OFFSET_ID, DEFAULT_RATE_LIMIT,
    DEFAULT_SESSION_NAME, DEFAULT_EXPORT_PATH, DEFAULT_DB_URL,
    DEFAULT_CHANNEL, DEFAULT_CHANNEL_PRIORITY, DEFAULT_MESSAGES_PER_MINUTE,
    DEFAULT_DELAY_BETWEEN_CHANNELS, DEFAULT_SESSION_COOLDOWN
)
from modules.telegram_collector import TelegramCollector, export_messages_to_file
from modules.models import ChannelConfig, RateLimitConfig


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
        messages_per_minute=rate_limit,
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
            
            # Collect messages from each channel
            channel_messages = {}  # Store messages per channel
            for channel in channel_list:
                if not channel.enabled:
                    continue
                    
                logger.info(f"Starting collection from channel: {channel.username}")
                try:
                    messages = await collector.collect_from_channel(channel, max_messages, offset_id)
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





@cli.command(name='reports')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--channels', '-c', multiple=True, help='Specific channels to generate reports for (default: all channels)')
@click.option('--output-dir', '-o', default='./reports/channels', help='Output directory for channel reports')
@click.help_option('-h', '--help')
def generate_reports(verbose: bool, channels: tuple, output_dir: str) -> None:
    """Generate comprehensive pandas-based reports for Telegram channels.
    
    This command creates detailed reports for each channel including:
    - Message statistics and metrics
    - Temporal analysis and trends
    - Content and media analysis
    - Engagement metrics
    - Data quality assessment
    - Export to multiple formats (CSV, JSON, Excel)
    
    Examples:
        python main.py reports                    # Generate reports for all channels
        python main.py reports -c @books         # Generate report for specific channel
        python main.py reports -c @books -c @books_magazine  # Multiple channels
        python main.py reports -o ./custom_reports  # Custom output directory
        python main.py reports -v                # Verbose logging
    """
    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    try:
        # Determine which channels to process
        if not channels:
            # Use default channels from config
            from config import OTHER_CHANNELS, DEFAULT_CHANNEL
            channels_to_process = [DEFAULT_CHANNEL] + OTHER_CHANNELS
            logger.info(f"📺 No channels specified, using default channels: {channels_to_process}")
        else:
            channels_to_process = list(channels)
            logger.info(f"📺 Generating reports for specified channels: {channels_to_process}")
        
        # Initialize channel reporter
        from modules.channel_reporter import ChannelReporter
        from modules.database_service import TelegramDBService
        
        db_service = TelegramDBService(DEFAULT_DB_URL)
        reporter = ChannelReporter(db_service, output_dir)
        
        logger.info(f"📊 Starting channel report generation...")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Generate reports for all specified channels
        results = asyncio.run(reporter.generate_all_channel_reports(channels_to_process))
        
        # Display results summary
        logger.info("=" * 60)
        logger.info("📋 CHANNEL REPORT GENERATION SUMMARY")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        errors = 0
        
        for channel, result in results.items():
            if result['status'] == 'success':
                successful += 1
                report = result['report']
                saved_files = report.get('saved_files', {})
                
                logger.info(f"✅ {channel}: SUCCESS")
                logger.info(f"   📊 Total Messages: {report.get('data_summary', {}).get('total_messages', 0):,}")
                logger.info(f"   📁 Files Generated:")
                for file_type, file_path in saved_files.items():
                    logger.info(f"     - {file_type.upper()}: {os.path.basename(file_path)}")
                
            elif result['status'] == 'failed':
                failed += 1
                logger.warning(f"⚠️  {channel}: FAILED - {result.get('error', 'Unknown error')}")
            else:
                errors += 1
                logger.error(f"❌ {channel}: ERROR - {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)
        logger.info(f"📈 SUMMARY: {successful} successful, {failed} failed, {errors} errors")
        logger.info(f"📁 All reports saved to: {output_dir}")
        
        if successful > 0:
            logger.info("🎉 Channel reports generated successfully!")
            logger.info("💡 Use the generated files for further analysis and visualization")
        
    except Exception as e:
        logger.error(f"❌ Channel report generation failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@cli.command(name='dashboard')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--channels', '-c', multiple=True, help='Specific channels to generate dashboards for (default: all channels)')
@click.option('--reports-dir', '-r', default='./reports/channels', help='Directory containing channel reports')
@click.option('--output-dir', '-o', default='./reports/dashboards', help='Output directory for generated dashboards')
@click.option('--template-dir', '-t', default='./templates', help='Directory containing Jinja2 templates')
@click.help_option('-h', '--help')
def generate_dashboards(verbose: bool, channels: tuple, reports_dir: str, output_dir: str, template_dir: str) -> None:
    """Generate interactive Plotly dashboards from channel reports.
    
    This command reads the JSON reports generated by the 'reports' command
    and creates interactive HTML dashboards using Plotly charts and Jinja2 templates.
    
    Examples:
        python main.py dashboard                    # Generate dashboards for all channels
        python main.py dashboard -c @books         # Generate dashboard for specific channel
        python main.py dashboard -c @books -c @books_magazine  # Multiple channels
        python main.py dashboard -r ./custom_reports  # Custom reports directory
        python main.py dashboard -o ./custom_dashboards  # Custom output directory
        python main.py dashboard -t ./custom_templates  # Custom template directory
        python main.py dashboard -v                # Verbose logging
    """
    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    try:
        # Determine which channels to process
        if not channels:
            # Scan reports directory to find available channels
            from pathlib import Path
            
            reports_path = Path(reports_dir)
            if not reports_path.exists():
                logger.error(f"❌ Reports directory not found: {reports_dir}")
                logger.error("Please run 'python main.py reports' first to generate channel reports")
                return
            
            # Find all channel directories
            channel_dirs = [d.name for d in reports_path.iterdir() if d.is_dir()]
            if not channel_dirs:
                logger.error(f"❌ No channel reports found in: {reports_dir}")
                logger.error("Please run 'python main.py reports' first to generate channel reports")
                return
            
            channels_to_process = channel_dirs
            logger.info(f"📺 No channels specified, found {len(channels_to_process)} channels with reports: {channels_to_process}")
        else:
            channels_to_process = list(channels)
            logger.info(f"📺 Generating dashboards for specified channels: {channels_to_process}")
        
        # Initialize dashboard generator
        from modules.dashboard_generator import ReportDashboardGenerator
        
        dashboard_gen = ReportDashboardGenerator(reports_dir, output_dir, template_dir)
        
        logger.info(f"📊 Starting dashboard generation...")
        logger.info(f"📁 Reports directory: {reports_dir}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📁 Template directory: {template_dir}")
        
        # Generate dashboards for all specified channels
        results = dashboard_gen.generate_all_dashboards(channels_to_process)
        
        # Display results summary
        logger.info("=" * 60)
        logger.info("📋 DASHBOARD GENERATION SUMMARY")
        logger.info("=" * 60)
        
        successful = 0
        failed = 0
        errors = 0
        
        for channel, result in results.items():
            if result['status'] == 'success':
                successful += 1
                dashboard_info = result['dashboard']
                
                logger.info(f"✅ {channel}: SUCCESS")
                logger.info(f"   📊 Dashboard: {dashboard_info.get('html_file', 'N/A')}")
                logger.info(f"   📁 Output: {dashboard_info.get('output_dir', 'N/A')}")
                
            elif result['status'] == 'failed':
                failed += 1
                logger.warning(f"⚠️  {channel}: FAILED - {result.get('error', 'Unknown error')}")
            else:
                errors += 1
                logger.error(f"❌ {channel}: ERROR - {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)
        logger.info(f"📈 SUMMARY: {successful} successful, {failed} failed, {errors} errors")
        logger.info(f"📁 All dashboards saved to: {output_dir}")
        
        if successful > 0:
            logger.info("🎉 Dashboards generated successfully!")
            logger.info("💡 Open the HTML files in your browser to view interactive charts")
            
            # Show index file if it exists
            index_file = os.path.join(output_dir, 'index.html')
            if os.path.exists(index_file):
                logger.info(f"🏠 Main dashboard index: {index_file}")
        
    except Exception as e:
        logger.error(f"❌ Dashboard generation failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    cli()
