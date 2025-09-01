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
from modules.combine_processor import (
    auto_detect_channels_from_raw_advanced,
    combine_existing_collections,
)
from modules.models import ChannelConfig, RateLimitConfig
from modules.file_report_processor import display_results_summary, process_channel_reports
from modules.db_report_processor import display_db_results_summary
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
    "--fax-fessages",
    "-f",
    default=DEFAULT_MAX_MESSAGES,
    type=int,
    help="Maximum messages to collect per channel (default: no limit)",
)
@click.option(
    "--offset-id",
    "-o",
    default=DEFAULT_OFFSET_ID,
    type=int,
    help="Start collecting from message ID greater than this (default: 0 for complete history)",
)
@click.option(
    "--rate-limit",
    "-r",
    default=DEFAULT_RATE_LIMIT,
    type=int,
    help="Messages per minute rate limit (default: 120)",
)
@click.option(
    "--session-name", "-s", default=DEFAULT_SESSION_NAME, help="Telegram session name"
)
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


@cli.command(name="report")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--file-messages", "-f", is_flag=True, help="Generate message analysis reports from combined files")
@click.option( "--db-messages", "-d", is_flag=True, help="Generate message analysis reports from database API endpoints",)
@click.help_option("-h", "--help")
def report(verbose: bool, file_messages: bool, db_messages: bool) -> None:
    """Generate reports for Telegram data analysis.
    This command can generate different types of reports:
    - Message analysis reports (when --file-messages flag is used)
    - Message analysis reports (when --db-messages flag is used)
    Examples:
        python main.py report --file-messages              # Generate message analysis reports from combined files
        python main.py report --db-messages              # Generate message analysis reports from database API endpoints
    """
    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.info("Verbose logging enabled")
    try:
        if file_messages:
            logger.info("📊 Generating message analysis reports from combined files...")
            # Process channel reports using the new module
            results = process_channel_reports(
                "reports/collections", "reports/messages"
            )
            # Display results summary
            display_results_summary(results)
        elif db_messages:
            logger.info("📊 Generating message analysis reports from database API endpoints...")
            # Process channel reports using the new module
            results = process_channel_reports(
                "reports/collections", "reports/messages"
            )
            # Display results summary
            display_results_summary(results)
        else:
            logger.info("No analysis type specified. Use --file-messages to generate message analysis reports.")
            return

            display_results_summary(results)
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        if verbose:
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        raise

        # Generate channel overview report
        async def generate_channel_report():
            async with db_service:
                # Get basic stats which include channel information
                stats = await db_service.get_stats()
                # Create channel overview report
                channel_report = {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": "channel_overview",
                    "generated_by": "main.py report channels",
                    "summary": {
                        "total_messages": stats.get("total_messages", 0),
                        "total_channels": stats.get("total_channels", 0),
                        "media_messages": stats.get("media_messages", 0),
                        "text_messages": stats.get("text_messages", 0),
                    },
                }
                return channel_report

        report = asyncio.run(generate_channel_report())
        # Save the report to a JSON file
        report_filename = os.path.join("./reports", "channels_overview.json")
        with open(report_filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(
            f"✅ Channel overview report generated successfully at: {report_filename}"
        )
        logger.info(f"📺 Total Channels: {report['summary']['total_channels']}")
        logger.info(f"� Total Messages: {report['summary']['total_messages']:,}")
        logger.info(f"📁 Media Messages: {report['summary']['media_messages']:,}")
        logger.info(f"📝 Text Messages: {report['summary']['text_messages']:,}")
    except Exception as e:
        logger.error(f"❌ Channel overview report generation failed: {e}")
        if verbose:
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@cli.command(name="dashboard")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable detailed logging and progress information",
)
@click.option(
    "--channels",
    "-c",
    multiple=True,
    help="Specific channels to process (e.g., @books, @books_magazine). If not specified, processes all available channels",
)
@click.option(
    "--reports-dir",
    "-r",
    default="./reports/channels",
    help="Source directory containing channel JSON reports (default: ./reports/channels)",
)
@click.option(
    "--output-dir",
    "-o",
    default="./reports/dashboards",
    help="Output directory for generated HTML dashboards (default: ./reports/dashboards)",
)
@click.option(
    "--template-dir",
    "-t",
    default="./templates",
    help="Directory containing Jinja2 HTML templates (default: ./templates)",
)
@click.help_option("-h", "--help")
def generate_dashboards(
    verbose: bool, channels: tuple, reports_dir: str, output_dir: str, template_dir: str
) -> None:
    """Generate interactive Plotly dashboards from channel reports.
    This command creates professional HTML dashboards with interactive charts from
    the JSON reports generated by 'python main.py report messages'.
    FEATURES:
        • Interactive Plotly charts (time series, pie charts, histograms)
        • Professional HTML templates with responsive design
        • Auto-discovery of available channel reports
        • Individual channel dashboards + main index dashboard
        • Google Analytics integration
    EXAMPLES:
        Basic Usage:
            python main.py dashboard                    # Generate dashboards for all channels
        Specific Channels:
            python main.py dashboard -c @books         # Single channel
            python main.py dashboard -c @books -c @books_magazine  # Multiple channels
        Custom Directories:
            python main.py dashboard -r ./custom_reports     # Custom reports location
            python main.py dashboard -o ./custom_output     # Custom output location
            python main.py dashboard -t ./custom_templates  # Custom template location
        Verbose Mode:
            python main.py dashboard -v                # Enable detailed logging
    OUTPUT:
        • Individual channel dashboards: ./reports/dashboards/{channel}_dashboard.html
        • Main index dashboard: ./reports/dashboards/index.html
        • All files are self-contained and can be shared/viewed in any browser
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
                logger.error(
                    "Please run 'python main.py report messages' first to generate channel reports"
                )
                return
            # Find all channel directories
            channel_dirs = [d.name for d in reports_path.iterdir() if d.is_dir()]
            if not channel_dirs:
                logger.error(f"❌ No channel reports found in: {reports_dir}")
                logger.error(
                    "Please run 'python main.py report messages' first to generate channel reports"
                )
                return
            channels_to_process = channel_dirs
            logger.info(
                f"📺 No channels specified, found {len(channels_to_process)} channels with reports: {channels_to_process}"
            )
        else:
            channels_to_process = list(channels)
            logger.info(
                f"📺 Generating dashboards for specified channels: {channels_to_process}"
            )
        # Initialize dashboard generator
        from modules.dashboard_generator import ReportDashboardGenerator

        dashboard_gen = ReportDashboardGenerator(reports_dir, output_dir, template_dir)
        logger.info("📊 Starting dashboard generation...")
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
            if result["status"] == "success":
                successful += 1
                dashboard_info = result["dashboard"]
                logger.info(f"✅ {channel}: SUCCESS")
                logger.info(
                    f"   📊 Dashboard: {dashboard_info.get('html_file', 'N/A')}"
                )
                logger.info(f"   📁 Output: {dashboard_info.get('output_dir', 'N/A')}")
            elif result["status"] == "failed":
                failed += 1
                logger.warning(
                    f"⚠️  {channel}: FAILED - {result.get('error', 'Unknown error')}"
                )
            else:
                errors += 1
                logger.error(
                    f"❌ {channel}: ERROR - {result.get('error', 'Unknown error')}"
                )
        logger.info("=" * 60)
        logger.info(
            f"📈 SUMMARY: {successful} successful, {failed} failed, {errors} errors"
        )
        logger.info(f"📁 All dashboards saved to: {output_dir}")
        if successful > 0:
            logger.info("🎉 Dashboards generated successfully!")
            logger.info(
                "💡 Open the HTML files in your browser to view interactive charts"
            )
            # Show index file if it exists
            index_file = os.path.join(output_dir, "index.html")
            if os.path.exists(index_file):
                logger.info(f"🏠 Main dashboard index: {index_file}")
    except Exception as e:
        logger.error(f"❌ Dashboard generation failed: {e}")
        if verbose:
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    cli()
