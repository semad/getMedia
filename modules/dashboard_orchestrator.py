"""
Dashboard Orchestrator for Telegram message analysis.

This service coordinates:
- StatsService: Data collection and calculation
- DashboardRenderer: HTML generation and rendering
- FileManager: File operations and I/O
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from modules.stats_service import StatsService
from modules.dashboard_renderer import DashboardRenderer
from modules.file_manager import FileManager
from modules.database_service import TelegramDBService
from config import DEFAULT_GA_MEASUREMENT_ID


class DashboardOrchestrator:
    """Orchestrates the complete dashboard generation process."""
    
    def __init__(self, output_dir: str = "./reports/html"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        from config import DEFAULT_DB_URL
        self.db_service = TelegramDBService(DEFAULT_DB_URL)
        self.stats_service = StatsService(self.db_service)
        self.renderer = DashboardRenderer("templates")
        self.file_manager = FileManager(output_dir)
        
        # Validate setup
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate that all services are properly configured."""
        try:
            # Validate output directory
            if not self.file_manager.validate_output_directory():
                raise RuntimeError(f"Output directory {self.output_dir} is not accessible")
            
            # Validate templates
            template_validation = self.renderer.validate_templates()
            failed_templates = [name for name, valid in template_validation.items() if not valid]
            
            if failed_templates:
                self.logger.warning(f"Some templates failed validation: {failed_templates}")
            
            self.logger.info("Dashboard orchestrator setup validated successfully")
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {e}")
            raise
    
    async def generate_complete_dashboard_system(self) -> Dict[str, Any]:
        """Generate the complete dashboard system."""
        try:
            self.logger.info("🚀 Starting complete dashboard generation...")
            
            # Step 1: Collect database statistics
            self.logger.info("📊 Collecting database statistics...")
            async with self.db_service:
                db_stats = await self.stats_service.get_database_stats()
                if not db_stats:
                    raise RuntimeError("Failed to collect database statistics")
            
            # Step 2: Generate all HTML content
            self.logger.info("🎨 Generating HTML content...")
            html_files = await self._generate_all_html_content(db_stats)
            
            # Step 3: Write all files
            self.logger.info("💾 Writing HTML files...")
            write_results = self.file_manager.write_multiple_files(html_files)
            
            # Step 4: Generate summary
            summary = self._generate_summary(db_stats, write_results)
            
            self.logger.info("✅ Dashboard generation completed successfully!")
            return summary
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            raise
    
    async def _generate_all_html_content(self, db_stats: Dict[str, Any]) -> List[tuple]:
        """Generate HTML content for all dashboards."""
        html_files = []
        
        # Generate index dashboard
        index_html = self.renderer.render_index_dashboard(
            db_stats, 
            DEFAULT_GA_MEASUREMENT_ID
        )
        html_files.append(("index.html", index_html))
        
        # Generate database analysis page
        db_analysis_html = self.renderer.render_database_analysis(
            db_stats, 
            DEFAULT_GA_MEASUREMENT_ID
        )
        html_files.append(("database_analysis.html", db_analysis_html))
        
        # Generate channel dashboards
        channel_breakdown = db_stats.get('channel_breakdown', {})
        for channel, message_count in channel_breakdown.items():
            try:
                # Get channel-specific stats
                channel_stats = await self.stats_service.get_channel_stats(
                    channel, 
                    message_count
                )
                
                # Generate channel dashboard HTML
                channel_html = self.renderer.render_channel_dashboard(
                    channel,
                    channel_stats,
                    DEFAULT_GA_MEASUREMENT_ID
                )
                
                # Create safe filename
                safe_filename = f"channel_{self.stats_service.safe_channel_name(channel)}.html"
                html_files.append((safe_filename, channel_html))
                
                self.logger.info(f"✅ Channel dashboard generated: {safe_filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate dashboard for channel {channel}: {e}")
                continue
        
        return html_files
    
    def _generate_summary(self, db_stats: Dict[str, Any], write_results: Dict[str, bool]) -> Dict[str, Any]:
        """Generate a summary of the dashboard generation process."""
        successful_files = [name for name, success in write_results.items() if success]
        failed_files = [name for name, success in write_results.items() if not success]
        
        # Get file information
        file_info = self.file_manager.list_generated_files()
        total_size = self.file_manager.get_directory_size()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'output_directory': self.output_dir,
            'database_stats': {
                'total_messages': db_stats.get('total_messages', 0),
                'total_channels': db_stats.get('total_channels', 0),
                'total_storage_gb': db_stats.get('total_storage_gb', 0.0)
            },
            'generation_results': {
                'total_files': len(write_results),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'success_rate': len(successful_files) / len(write_results) if write_results else 0
            },
            'file_details': {
                'successful_files': successful_files,
                'failed_files': failed_files,
                'file_info': file_info,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        }
    
    def cleanup_old_dashboards(self, max_age_hours: int = 24) -> int:
        """Clean up old dashboard files."""
        try:
            cleaned_count = self.file_manager.cleanup_old_files(
                pattern="*.html", 
                max_age_hours=max_age_hours
            )
            
            if cleaned_count > 0:
                self.logger.info(f"🧹 Cleaned up {cleaned_count} old dashboard files")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get current status of the dashboard system."""
        try:
            file_info = self.file_manager.list_generated_files()
            total_size = self.file_manager.get_directory_size()
            
            return {
                'output_directory': self.output_dir,
                'total_files': len(file_info),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'files': file_info,
                'last_updated': max([f['modified'] for f in file_info]) if file_info else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard status: {e}")
            return {}
    
    def validate_templates(self) -> Dict[str, bool]:
        """Validate all templates."""
        return self.renderer.validate_templates()
    
    def test_rendering(self) -> Dict[str, bool]:
        """Test rendering with sample data."""
        try:
            sample_stats = {
                'total_messages': 1000,
                'total_channels': 2,
                'total_storage_gb': 10.5,
                'media_breakdown': {'document': 800, 'photo': 200},
                'channel_breakdown': {'@test1': 600, '@test2': 400},
                'date_range': {'start': '2020-01-01', 'end': '2025-01-01'}
            }
            
            # Test index rendering
            index_html = self.renderer.render_index_dashboard(
                sample_stats, 
                DEFAULT_GA_MEASUREMENT_ID
            )
            index_success = len(index_html) > 100
            
            # Test channel rendering
            channel_stats = {
                'channel': '@test1',
                'message_count': 600,
                'media_count': 480,
                'text_count': 120,
                'charts': {
                    'dates': ['2020-01', '2020-02'],
                    'message_counts': [300, 300],
                    'file_sizes': [5.2, 3.8],
                    'hours': [0, 1],
                    'hourly_counts': [25, 25],
                    'media_types': ['document', 'photo'],
                    'media_counts': [400, 80]
                }
            }
            
            channel_html = self.renderer.render_channel_dashboard(
                '@test1',
                channel_stats,
                DEFAULT_GA_MEASUREMENT_ID
            )
            channel_success = len(channel_html) > 100
            
            return {
                'index_rendering': index_success,
                'channel_rendering': channel_success,
                'overall_success': index_success and channel_success
            }
            
        except Exception as e:
            self.logger.error(f"Rendering test failed: {e}")
            return {'overall_success': False, 'error': str(e)}
