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
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.import_processor import run_import
from modules.telegram_collector import TelegramCollector, DatabaseChecker
from modules.telegram_exporter import TelegramMessageExporter
from modules.telegram_analyzer import TelegramDataAnalyzer
from modules.models import ChannelConfig, RateLimitConfig
from modules.database_service import TelegramDBService


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


@cli.command(name='import')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--batch-size', '-b', default=100, type=int, help='Batch size for importing messages (default: 100)')
@click.option('--dry-run', is_flag=True, help='Show what would be imported without actually importing')
@click.option('--skip-duplicates', is_flag=True, help='Skip messages that already exist in database')
@click.option('--validate-only', is_flag=True, help='Only validate data format and quality without importing')
@click.option('--check-quality', is_flag=True, help='Check data quality and report potential issues before import')
@click.option('--max-retries', default=5, type=int, help='Maximum number of retry attempts for failed operations (default: 5)')
@click.option('--retry-delay', default=1.0, type=float, help='Base delay between retries in seconds (default: 1.0)')
@click.option('--max-delay', default=60.0, type=float, help='Maximum delay between retries in seconds (default: 60.0)')
@click.option('--batch-delay', default=1.0, type=float, help='Delay between batches in seconds (default: 1.0)')
@click.option('--limit', '-l', default=None, type=int, help='Limit the number of records to import (default: no limit)')
@click.help_option('-h', '--help')
def import_cmd(data_file, verbose, batch_size, dry_run, skip_duplicates, validate_only, check_quality, max_retries, retry_delay, max_delay, batch_delay, limit):
    """Import, validate, and check quality of Telegram message data from JSON or CSV file.
    
    This unified command handles:
    - Data validation (format and structure)
    - Data quality checking (JSON compatibility issues)
    - Data import into database
    
    Use --validate-only to only validate without importing.
    Use --check-quality to identify potential import issues.
    Use --dry-run to see what would be imported without actually importing.
    Use --limit to process only a subset of records for testing.
    """
    setup_logging(verbose)
    
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info("Verbose logging enabled")
    
    logger = logging.getLogger(__name__)
    
    if validate_only:
        logger.info(f"VALIDATION MODE - Validating data from: {data_file}")
    elif dry_run:
        logger.info(f"DRY RUN MODE - Would import data from: {data_file}")
    else:
        logger.info(f"IMPORT MODE - Importing data from: {data_file}")
    
    # Database URL - you can make this configurable
    db_url = "http://localhost:8000"
    
    if not validate_only:
        logger.info(f"Connecting to database at: {db_url}")
    
    # Run the import/validation
    try:
        stats = asyncio.run(run_import(
            data_file=data_file,
            db_url=db_url,
            batch_size=batch_size,
            dry_run=dry_run,
            skip_duplicates=skip_duplicates,
            validate_only=validate_only,
            check_quality=check_quality,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_delay=max_delay,
            batch_delay=batch_delay,
            limit=limit
        ))
        
        # Final summary based on mode
        if validate_only:
            logger.info("VALIDATION COMPLETE")
            if check_quality:
                logger.info("Data quality check completed - review any warnings above")
            logger.info(f"Summary: {stats.total_messages} total messages processed")
        elif dry_run:
            logger.info(f"DRY RUN COMPLETE - Would import {stats.imported_count} messages")
        else:
            logger.info(f"IMPORT COMPLETE - Imported {stats.imported_count} messages")
        
        # Show detailed statistics
        if not validate_only:
            logger.info(f"Summary: {stats.imported_count} imported, {stats.skipped_count} skipped, {stats.error_count} errors")
            logger.info(f"Retry statistics: {stats.retry_count} total retries performed")
            
            if stats.duration:
                logger.info(f"Total time: {stats.duration:.1f} seconds")
                if stats.total_messages > 0:
                    rate = stats.imported_count / stats.duration
                    logger.info(f"Import rate: {rate:.1f} messages/second")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise click.Abort()


@cli.command()
@click.option('--channels', '-c', 
              help='Comma-separated list of channel usernames')
@click.option('--max-messages', '-m', default=None, type=int,
              help='Maximum messages to collect per channel (default: no limit)')
@click.option('--offset-id', '-o', default=0, type=int,
              help='Start collecting from message ID greater than this (default: 0 for complete history)')
@click.option('--rate-limit', '-r', default=120, type=int,
              help='Messages per minute rate limit (default: 120)')
@click.option('--session-name', '-s', default='telegram_collector',
              help='Telegram session name')
@click.option('--dry-run', is_flag=True,
              help='Run without storing to database')
@click.option('--export', '-e', default=None, help='Export collected messages to JSON file')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging output')
@click.help_option('-h', '--help')
def collect(channels, max_messages, offset_id, rate_limit, session_name, dry_run, export, verbose):
    """Collect messages from Telegram channels."""
    
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    logger.info("💡 Tip: Press Ctrl+C to gracefully stop collection and save progress")
    
    # Load configuration
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    
    if not api_id or not api_hash:
        logger.error("Missing TG_API_ID or TG_API_HASH environment variables")
        logger.error("Please set these environment variables to use the collect command")
        return
    
    # Parse channels
    if channels:
        channel_list = [ChannelConfig(username=ch.strip()) for ch in channels.split(',')]
    else:
        channel_list = [
            ChannelConfig("@SherwinVakiliLibrary", priority=1),
        ]
    
    # Configure rate limiting (default to maximum speed)
    rate_config = RateLimitConfig(
        messages_per_minute=rate_limit if rate_limit != 120 else 1000,  # Default to 1000 if not specified
        delay_between_channels=1,  # Minimal delay between channels
        session_cooldown=60  # Reduced cooldown for faster collection
    )
    
    # Global variables for signal handling
    collector_instance = None
    all_messages = []
    task = None
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n🛑 Interrupt signal received (Ctrl+C)")
        logger.info("🔄 Gracefully shutting down...")
        if task and not task.done():
            logger.info("📝 Cancelling collection task and saving progress...")
            task.cancel()
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    async def run_collector():
        nonlocal collector_instance, all_messages
        
        collector_instance = TelegramCollector(rate_config)
        
        try:
            # Check database for message IDs
            actual_offset_id = offset_id
            if offset_id == 0:
                logger.info("Checking database for message IDs...")
                async with DatabaseChecker(db_url) as checker:
                    for channel in channel_list:
                        if channel.enabled:
                            highest_id = await checker.get_last_message_id(channel.username)
                            if highest_id:
                                actual_offset_id = highest_id
                                logger.info(f"Found highest message ID in database: {highest_id}")
            
            # Initialize database service
            if not dry_run:
                collector_instance.db_service = TelegramDBService(db_url)
                async with collector_instance.db_service:
                    if await collector_instance.initialize(api_id, api_hash, session_name):
                        for channel in channel_list:
                            if channel.enabled:
                                logger.info(f"Starting collection from channel: {channel.username}")
                                try:
                                    messages = await collector_instance.collect_from_channel(channel, max_messages, actual_offset_id)
                                    all_messages.extend(messages)
                                except asyncio.CancelledError:
                                    logger.info("🛑 Collection cancelled, stopping...")
                                    break
                        await collector_instance.close()
            else:
                logger.info("DRY RUN MODE - No database operations")
                if await collector_instance.initialize(api_id, api_hash, session_name):
                    for channel in channel_list:
                        if channel.enabled:
                            logger.info(f"Starting collection from channel: {channel.username}")
                            try:
                                messages = await collector_instance.collect_from_channel(channel, max_messages, actual_offset_id)
                                all_messages.extend(messages)
                            except asyncio.CancelledError:
                                logger.info("🛑 Collection cancelled, stopping...")
                                # Even if cancelled, we might have some messages
                                if messages:
                                    all_messages.extend(messages)
                                break
                    await collector_instance.close()
                    
        except asyncio.CancelledError:
            logger.info("🛑 Collection task was cancelled")
        except Exception as e:
            logger.error(f"Error during collection: {e}")
        finally:
            # Always try to close the collector
            if collector_instance:
                try:
                    await collector_instance.close()
                except Exception as e:
                    logger.warning(f"Warning: Could not close collector cleanly: {e}")
        
        # Print statistics
        logger.info(f"📝 Collection completed: {len(all_messages)} messages")
            
        if collector_instance:
            logger.info(f"Total messages processed: {collector_instance.stats.get('total_messages', len(all_messages))}")
            logger.info(f"Channels processed: {collector_instance.stats.get('channels_processed', len([ch for ch in channel_list if ch.enabled]))}")
            logger.info(f"Errors encountered: {collector_instance.stats.get('errors', 0)}")
        else:
            logger.info(f"Total messages collected: {len(all_messages)}")
            logger.info(f"Channels processed: {len([ch for ch in channel_list if ch.enabled])}")
            logger.info("Errors encountered: 0")
        
    # Create and run the collection task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        task = loop.create_task(run_collector())
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received in main loop")
        if task and not task.done():
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
    
    # Export to JSON file if requested (this runs regardless of interruption)
    if export:
        try:
            # Get stats safely
            stats = collector_instance.stats if collector_instance else {
                'total_messages': len(all_messages),
                'channels_processed': len([ch for ch in channel_list if ch.enabled]),
                'errors': 0
            }
            
            # Try to get messages from collector if all_messages is empty
            if all_messages:
                messages_to_export = all_messages
                total_count = len(all_messages)
            elif collector_instance:
                # Get messages that were collected before interruption
                collected_messages = collector_instance.get_collected_messages()
                if collected_messages:
                    messages_to_export = collected_messages
                    total_count = len(collected_messages)
                    logger.info(f"📝 Retrieved {total_count} messages from collector instance")
                else:
                    messages_to_export = []
                    total_count = stats.get('total_messages', 0)
                    logger.info(f"📝 No messages in all_messages or collector, but stats show {total_count} processed")
            else:
                # Create placeholder messages based on stats
                messages_to_export = []
                total_count = stats.get('total_messages', 0)
                logger.info(f"📝 No messages in all_messages, but stats show {total_count} processed")
            
            if total_count > 0:
                export_data = {
                    'metadata': {
                        'collected_at': datetime.now().isoformat(),
                        'channels': [ch.username for ch in channel_list if ch.enabled],
                        'total_messages': total_count,
                        'collection_stats': stats,
                        'cancelled': task and task.cancelled() if task else False,
                        'note': 'Messages may be incomplete due to interruption' if task and task.cancelled() else None
                    },
                    'messages': messages_to_export
                }
                
                # Ensure the export filename has .json extension
                export_filename = export
                if not export_filename.endswith('.json'):
                    export_filename = f"{export_filename}.json"
                
                with open(export_filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
                
                logger.info(f"✅ Exported {total_count} messages to: {export_filename}")
            else:
                logger.warning("No messages to export")
            
        except Exception as e:
            logger.error(f"Failed to export messages: {e}")
    
    # Print final statistics
    if task and task.cancelled():
        logger.info("🛑 Collection was cancelled by user")
        logger.info(f"📝 Partial collection completed: {len(all_messages)} messages")
    else:
        logger.info("Collection completed!")
        
    if collector_instance:
        logger.info(f"Total messages processed: {collector_instance.stats.get('total_messages', len(all_messages))}")
        logger.info(f"Channels processed: {collector_instance.stats.get('channels_processed', len([ch for ch in channel_list if ch.enabled]))}")
        logger.info(f"Errors encountered: {collector_instance.stats.get('errors', 0)}")
    else:
        logger.info(f"Total messages collected: {len(all_messages)}")
        logger.info(f"Channels processed: {len([ch for ch in channel_list if ch.enabled])}")
        logger.info("Errors encountered: 0")
    
    if export and all_messages:
        logger.info(f"📁 Messages saved to: {export_filename}")
    
    loop.close()


@cli.command()
@click.option('--output', '-o', default='telegram_messages_export',
              help='Output filename without extension (default: telegram_messages_export)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'excel', 'all']), default='json',
              help='Export format: json, csv, excel, or all (default: json)')
@click.option('--batch-size', '-b', default=1000, type=int,
              help='Batch size for fetching messages (default: 1000)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--summary', '-s', is_flag=True, help='Generate and display summary report')
@click.help_option('-h', '--help')
def export(output, format, batch_size, verbose, summary):
    """Export all Telegram messages from database."""
    
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    # Get database URL from environment
    db_url = os.getenv('TELEGRAM_DB_URL', 'http://localhost:8000')
    logger.info(f"Connecting to database at: {db_url}")
    
    async def run_export():
        async with TelegramMessageExporter(db_url) as exporter:
            try:
                # Get total count first
                total_count = await exporter.get_total_count()
                logger.info(f"Total messages in database: {total_count}")
                
                if total_count == 0:
                    logger.warning("No messages found in database")
                    return
                
                # Fetch messages
                logger.info("Exporting all messages from database")
                messages = await exporter.get_all_messages(batch_size)
                
                if not messages:
                    logger.warning("No messages fetched")
                    return
                
                logger.info(f"Successfully fetched {len(messages)} messages")
                
                # Convert to pandas DataFrame
                df = exporter.create_dataframe(messages)
                
                # Generate summary report if requested
                if summary:
                    report = exporter.generate_summary_report(df)
                    print("\n" + report)
                
                # Export based on format
                success = False
                if format == 'json' or format == 'all':
                    json_filename = f"{output}.json"
                    success = exporter.export_to_json(df, json_filename)
                
                if format == 'csv' or format == 'all':
                    csv_filename = f"{output}.csv"
                    success = exporter.export_to_csv(df, csv_filename)
                
                if format == 'excel' or format == 'all':
                    excel_filename = f"{output}.xlsx"
                    success = exporter.export_to_excel(df, excel_filename)
                
                if success:
                    logger.info(f"Export completed successfully!")
                    if format == 'all':
                        logger.info(f"Output files: {output}.json, {output}.csv, {output}.xlsx")
                    else:
                        logger.info(f"Output file: {output}.{format}")
                    logger.info(f"Total messages exported: {len(messages)}")
                    
                    # Show DataFrame info
                    logger.info(f"DataFrame shape: {df.shape}")
                    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                else:
                    logger.error("Export failed!")
                    
            except Exception as e:
                logger.error(f"Export failed with error: {e}")
                raise click.Abort()
    
    asyncio.run(run_export())


@cli.command()
@click.argument('data_file', type=click.Path(exists=True), required=False)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--dashboard', '-d', is_flag=True, help='Generate interactive HTML dashboard')
@click.option('--output', '-o', default='telegram_analysis_dashboard.html',
              help='Output HTML file path for dashboard (default: telegram_analysis_dashboard.html)')
@click.option('--database', '-db', is_flag=True, help='Analyze database statistics instead of file data')
@click.option('--summary', '-s', is_flag=True, help='Generate summary report')
@click.help_option('-h', '--help')
def analyze(data_file, verbose, dashboard, output, database, summary):
    """Analyze Telegram message data from files or database with comprehensive reports.
    
    This unified command handles:
    - File analysis: Analyze data from JSON/CSV files with detailed reports
    - Database analysis: Show database statistics and summary information
    - Interactive dashboard generation (planned feature)
    
    Use --database to analyze database statistics instead of file data.
    Use --summary to generate concise summary reports.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    if database:
        # Database analysis mode
        logger.info("DATABASE ANALYSIS MODE - Analyzing database statistics")
        analyze_database(logger, summary)
    elif data_file:
        # File analysis mode
        logger.info(f"FILE ANALYSIS MODE - Analyzing data from: {data_file}")
        analyze_file(data_file, logger, dashboard, output, summary)
    else:
        # No arguments provided - show help
        logger.error("Please specify either a data file or use --database flag")
        logger.error("Examples:")
        logger.error("  python main.py analyze data.json          # Analyze file")
        logger.error("  python main.py analyze --database         # Analyze database")
        logger.error("  python main.py analyze --help             # Show help")


def analyze_database(logger, summary):
    """Analyze database statistics."""
    logger.info("Fetching database statistics...")
    
    # This would connect to the database and show stats
    # For now, show a placeholder with enhanced information
    logger.info("Database analysis feature - Enhanced implementation planned")
    logger.info("Will include:")
    logger.info("  - Total message count")
    logger.info("  - Channel breakdown")
    logger.info("  - Media type statistics")
    logger.info("  - Temporal analysis")
    logger.info("  - Storage usage")
    
    if summary:
        logger.info("Summary mode enabled - will show concise database overview")


def analyze_file(data_file, logger, dashboard, output, summary):
    """Analyze data from a file."""
    analyzer = TelegramDataAnalyzer(data_file)
    
    if analyzer.df is not None:
        if summary:
            # Generate concise summary
            basic_stats = analyzer.generate_basic_stats()
            logger.info("=== CONCISE SUMMARY ===")
            logger.info(f"Total Messages: {basic_stats.get('total_messages', 0)}")
            logger.info(f"Unique Channels: {basic_stats.get('unique_channels', 0)}")
            if 'date_range' in basic_stats:
                logger.info(f"Date Range: {basic_stats['date_range']['start']} to {basic_stats['date_range']['end']}")
        else:
            # Generate comprehensive report
            report = analyzer.generate_comprehensive_report()
            print("\n" + report)
        
        if dashboard:
            logger.info("Dashboard generation not yet implemented")
            logger.info("Use the console reports above for data analysis")
    else:
        logger.error("Failed to load data for analysis")


class DummyContext:
    """Dummy context manager for dry-run mode."""
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == '__main__':
    cli()
