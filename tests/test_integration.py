"""
Integration tests for the complete workflow: collect -> import -> reports -> dashboard.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from modules.telegram_collector import TelegramCollector
from modules.import_processor import run_import
from modules.channel_reporter import ChannelReporter
from modules.dashboard_generator import ReportDashboardGenerator


class TestCompleteWorkflow:
    """Test the complete workflow from collection to dashboard generation."""
    
    @pytest.fixture
    def workflow_setup(self, temp_dir):
        """Set up the complete workflow test environment."""
        # Create directory structure
        collections_dir = temp_dir / "collections"
        reports_dir = temp_dir / "reports"
        dashboards_dir = temp_dir / "dashboards"
        templates_dir = temp_dir / "templates"
        
        collections_dir.mkdir()
        reports_dir.mkdir()
        dashboards_dir.mkdir()
        templates_dir.mkdir()
        
        # Create sample templates
        self._create_workflow_templates(templates_dir)
        
        return {
            'temp_dir': temp_dir,
            'collections_dir': str(collections_dir),
            'reports_dir': str(reports_dir),
            'dashboards_dir': str(dashboards_dir),
            'templates_dir': str(templates_dir)
        }
    
    def _create_workflow_templates(self, templates_dir):
        """Create templates needed for the workflow."""
        # Channel dashboard template
        channel_template = """
        <!DOCTYPE html>
        <html>
        <head><title>{{ channel_name }} Dashboard</title></head>
        <body>
            <h1>{{ channel_name }} Dashboard</h1>
            <div id="time_series_chart">{{ time_series_chart | safe }}</div>
            <div id="media_pie_chart">{{ media_pie_chart | safe }}</div>
            <div id="file_size_histogram">{{ file_size_histogram | safe }}</div>
        </body>
        </html>
        """
        
        with open(templates_dir / "dashboard_channel.html", "w") as f:
            f.write(channel_template)
        
        # Index dashboard template
        index_template = """
        <!DOCTYPE html>
        <html>
        <head><title>Dashboard Index</title></head>
        <body>
            <h1>Dashboard Index</h1>
            <div id="overview_chart">{{ overview_chart | safe }}</div>
            <div id="channel_cards">{{ channel_cards | safe }}</div>
        </body>
        </html>
        """
        
        with open(templates_dir / "dashboard_index.html", "w") as f:
            f.write(index_template)
    
    @pytest.fixture
    def sample_workflow_data(self):
        """Create sample data for the workflow."""
        return {
            'messages': [
                {
                    'message_id': 1,
                    'channel_username': '@test_channel',
                    'text': 'First test message with some content',
                    'date': '2024-01-01T10:00:00',
                    'media_type': 'text',
                    'file_name': None,
                    'file_size': None,
                    'mime_type': None,
                    'views': 15,
                    'forwards': 2,
                    'replies': 1
                },
                {
                    'message_id': 2,
                    'channel_username': '@test_channel',
                    'text': 'Second message with document',
                    'date': '2024-01-01T11:00:00',
                    'media_type': 'document',
                    'file_name': 'test.pdf',
                    'file_size': 1024,
                    'mime_type': 'application/pdf',
                    'views': 25,
                    'forwards': 3,
                    'replies': 2
                },
                {
                    'message_id': 3,
                    'channel_username': '@test_channel',
                    'text': 'Third message with photo',
                    'date': '2024-01-01T12:00:00',
                    'media_type': 'photo',
                    'file_name': 'test.jpg',
                    'file_size': 512,
                    'mime_type': 'image/jpeg',
                    'views': 30,
                    'forwards': 1,
                    'replies': 0
                }
            ],
            'channels': ['@test_channel']
        }
    
    @pytest.mark.asyncio
    async def test_workflow_step1_collect(self, workflow_setup, sample_workflow_data):
        """Test step 1: Message collection and export."""
        from modules.models import ChannelConfig
        
        # Mock the Telegram collector
        with patch('telethon.TelegramClient') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(return_value=True)
            mock_client.iter_messages = AsyncMock(return_value=[])
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Create collector
            rate_config = Mock()
            rate_config.messages_per_minute = 30
            collector = TelegramCollector(rate_config)
            
            # Mock the client
            collector.client = mock_client
            
            # Simulate message collection
            collected_messages = sample_workflow_data['messages']
            collector.collected_messages = collected_messages
            
            # Export messages
            channel_list = [ChannelConfig(ch) for ch in sample_workflow_data['channels']]
            export_file = collector.export_messages_to_file(
                collected_messages,
                f"{workflow_setup['collections_dir']}/workflow_collection",
                channel_list,
                Mock()
            )
            
            # Verify export
            assert export_file is not None
            assert Path(export_file).exists()
            
            # Verify file content
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            assert data['metadata']['total_messages'] == 3
            assert data['metadata']['channels'] == ['@test_channel']
            assert data['metadata']['data_format'] == 'structured_dataframe'
            assert len(data['messages']) == 3
            
            return export_file
    
    @pytest.mark.asyncio
    async def test_workflow_step2_import(self, workflow_setup, sample_workflow_data):
        """Test step 2: Import messages to database."""
        # Create a collection file first
        collection_file = f"{workflow_setup['collections_dir']}/workflow_collection.json"
        
        collection_data = {
            'metadata': {
                'collected_at': '2024-01-01T10:00:00',
                'channels': sample_workflow_data['channels'],
                'total_messages': len(sample_workflow_data['messages']),
                'data_format': 'structured_dataframe',
                'fields': list(sample_workflow_data['messages'][0].keys())
            },
            'messages': sample_workflow_data['messages']
        }
        
        with open(collection_file, 'w') as f:
            json.dump(collection_data, f)
        
        # Mock the database service
        with patch('modules.import_processor.TelegramDBService') as mock_db_class:
            mock_db_service = Mock()
            mock_db_service.check_connection = AsyncMock(return_value=True)
            mock_db_service.store_message = AsyncMock(return_value=True)
            mock_db_class.return_value = mock_db_service
            
            # Run import
            result = await run_import(
                collection_file,
                "http://localhost:8000",
                dry_run=True  # Don't actually import for testing
            )
            
            # Verify import results
            assert result['total_messages'] == 3
            assert result['imported_count'] == 3
            assert result['error_count'] == 0
            assert result['start_time'] is not None
            assert result['end_time'] is not None
    
    @pytest.mark.asyncio
    async def test_workflow_step3_reports(self, workflow_setup, sample_workflow_data):
        """Test step 3: Generate channel reports."""
        # Mock the database service
        mock_db_service = Mock()
        mock_db_service.fetch_all_messages = AsyncMock(return_value=sample_workflow_data['messages'])
        
        # Create channel reporter
        reporter = ChannelReporter(mock_db_service)
        reporter.reports_dir = workflow_setup['reports_dir']
        
        # Generate report for the channel
        result = await reporter.generate_channel_report('@test_channel')
        
        # Verify report generation
        assert result['status'] == 'success'
        assert result['channel'] == '@test_channel'
        assert result['report_info'] is not None
        
        # Check if report files were created
        channel_report_dir = Path(workflow_setup['reports_dir']) / '@test_channel'
        assert channel_report_dir.exists()
        
        # Check for JSON report
        json_files = list(channel_report_dir.glob('*.json'))
        assert len(json_files) > 0
        
        # Verify report content
        with open(json_files[0], 'r') as f:
            report_data = json.load(f)
        
        assert 'channel_info' in report_data
        assert 'media_analysis' in report_data
        assert 'temporal_analysis' in report_data
        assert 'engagement_analysis' in report_data
        assert 'file_analysis' in report_data
    
    @pytest.mark.asyncio
    async def test_workflow_step4_dashboard(self, workflow_setup, sample_workflow_data):
        """Test step 4: Generate dashboards from reports."""
        # First create a sample report
        report_data = {
            'channel_info': {
                'total_messages': 3,
                'channel_username': '@test_channel',
                'date_range': {'start': '2024-01-01', 'end': '2024-01-01'},
                'active_days': 1
            },
            'media_analysis': {
                'total_media': 2,
                'media_percentage': 66.67,
                'media_breakdown': {'text': 1, 'document': 1, 'photo': 1}
            },
            'temporal_analysis': {
                'monthly_activity': {'2024-01': 3},
                'daily_activity': {'2024-01-01': 3},
                'hourly_activity': {'10': 1, '11': 1, '12': 1}
            },
            'engagement_analysis': {
                'total_views': 70,
                'total_forwards': 6,
                'total_replies': 3
            },
            'file_analysis': {
                'total_files': 2,
                'total_storage_mb': 1.5,
                'avg_file_size_mb': 0.75
            }
        }
        
        # Create report directory and file
        report_dir = Path(workflow_setup['reports_dir']) / '@test_channel'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / 'report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        # Create dashboard generator
        generator = ReportDashboardGenerator(
            workflow_setup['reports_dir'],
            workflow_setup['dashboards_dir'],
            workflow_setup['templates_dir']
        )
        
        # Generate dashboards
        results = generator.generate_all_dashboards(['@test_channel'])
        
        # Verify dashboard generation
        assert len(results) == 1
        assert results['@test_channel']['status'] == 'success'
        
        # Check if dashboard files were created
        dashboard_dir = Path(workflow_setup['dashboards_dir'])
        assert dashboard_dir.exists()
        
        # Check for channel dashboard
        channel_dashboard = dashboard_dir / '@test_channel'
        assert channel_dashboard.exists()
        
        # Check for index dashboard
        index_dashboard = dashboard_dir / 'index.html'
        assert index_dashboard.exists()
        
        # Verify dashboard content
        with open(index_dashboard, 'r') as f:
            content = f.read()
            assert 'Dashboard Index' in content
            assert 'overview_chart' in content
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, workflow_setup, sample_workflow_data):
        """Test the complete workflow integration."""
        # Step 1: Collect and export
        collection_file = await self.test_workflow_step1_collect(workflow_setup, sample_workflow_data)
        
        # Step 2: Import
        await self.test_workflow_step2_import(workflow_setup, sample_workflow_data)
        
        # Step 3: Generate reports
        await self.test_workflow_step3_reports(workflow_setup, sample_workflow_data)
        
        # Step 4: Generate dashboards
        await self.test_workflow_step4_dashboard(workflow_setup, sample_workflow_data)
        
        # Verify final output structure
        temp_dir = Path(workflow_setup['temp_dir'])
        
        # Check collections
        collections_dir = temp_dir / "collections"
        assert collections_dir.exists()
        collection_files = list(collections_dir.glob('*.json'))
        assert len(collection_files) > 0
        
        # Check reports
        reports_dir = temp_dir / "reports"
        assert reports_dir.exists()
        channel_reports = list(reports_dir.glob('@*'))
        assert len(channel_reports) > 0
        
        # Check dashboards
        dashboards_dir = temp_dir / "dashboards"
        assert dashboards_dir.exists()
        dashboard_files = list(dashboards_dir.rglob('*.html'))
        assert len(dashboard_files) > 0
        
        # Verify data consistency across workflow
        with open(collection_files[0], 'r') as f:
            collection_data = json.load(f)
        
        with open(channel_reports[0] / 'report.json', 'r') as f:
            report_data = json.load(f)
        
        # Verify message count consistency
        assert collection_data['metadata']['total_messages'] == report_data['channel_info']['total_messages']
        
        # Verify channel consistency
        assert collection_data['metadata']['channels'][0] == report_data['channel_info']['channel_username']
        
        print(f"✅ Complete workflow test passed!")
        print(f"   📁 Collections: {len(collection_files)} files")
        print(f"   📊 Reports: {len(channel_reports)} channels")
        print(f"   🎨 Dashboards: {len(dashboard_files)} HTML files")


class TestWorkflowErrorHandling:
    """Test error handling in the workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_with_invalid_data(self, temp_dir):
        """Test workflow behavior with invalid data."""
        # Test with empty messages
        empty_messages = []
        
        # This should not crash but handle gracefully
        from modules.telegram_collector import export_messages_to_file
        from modules.models import ChannelConfig
        
        channel_list = [ChannelConfig('@test_channel')]
        
        export_file = export_messages_to_file(
            empty_messages,
            f"{temp_dir}/empty_collection",
            channel_list,
            Mock()
        )
        
        assert export_file is not None
        assert Path(export_file).exists()
        
        # Verify empty collection file
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        assert data['metadata']['total_messages'] == 0
        assert len(data['messages']) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_malformed_data(self, temp_dir):
        """Test workflow behavior with malformed data."""
        # Test with malformed messages
        malformed_messages = [
            {'message_id': 1},  # Missing required fields
            {'text': 'Invalid message'},  # Missing message_id
            None,  # None message
            'not_a_dict'  # Wrong type
        ]
        
        from modules.telegram_collector import export_messages_to_file
        from modules.models import ChannelConfig
        
        channel_list = [ChannelConfig('@test_channel')]
        
        # This should handle malformed data gracefully
        export_file = export_messages_to_file(
            malformed_messages,
            f"{temp_dir}/malformed_collection",
            channel_list,
            Mock()
        )
        
        assert export_file is not None
        assert Path(export_file).exists()
        
        # Verify the export handled malformed data
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        # Should still create a valid file structure
        assert 'metadata' in data
        assert 'messages' in data
