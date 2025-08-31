"""
Unit tests for the dashboard generator module.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from modules.dashboard_generator import ReportDashboardGenerator


class TestReportDashboardGenerator:
    """Test ReportDashboardGenerator class."""
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create a ReportDashboardGenerator instance for testing."""
        reports_dir = temp_dir / "reports"
        output_dir = temp_dir / "dashboards"
        template_dir = temp_dir / "templates"
        
        # Create directories
        reports_dir.mkdir()
        output_dir.mkdir()
        template_dir.mkdir()
        
        # Create sample template files
        self._create_sample_templates(template_dir)
        
        return ReportDashboardGenerator(str(reports_dir), str(output_dir), str(template_dir))
    
    def _create_sample_templates(self, template_dir):
        """Create sample template files for testing."""
        # Create dashboard_channel.html template
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
        
        with open(template_dir / "dashboard_channel.html", "w") as f:
            f.write(channel_template)
        
        # Create dashboard_index.html template
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
        
        with open(template_dir / "dashboard_index.html", "w") as f:
            f.write(index_template)
    
    @pytest.fixture
    def sample_channel_report(self, temp_dir):
        """Create a sample channel report for testing."""
        report_data = {
            'channel_info': {
                'total_messages': 100,
                'channel_username': '@test_channel',
                'date_range': {'start': '2024-01-01', 'end': '2024-01-31'},
                'active_days': 31
            },
            'media_analysis': {
                'total_media': 30,
                'media_percentage': 30.0,
                'media_breakdown': {'text': 70, 'document': 20, 'photo': 10}
            },
            'temporal_analysis': {
                'monthly_activity': {'2024-01': 100},
                'daily_activity': {'2024-01-01': 3, '2024-01-02': 4},
                'hourly_activity': {'10': 15, '11': 20, '12': 25}
            },
            'engagement_analysis': {
                'total_views': 1000,
                'total_forwards': 50,
                'total_replies': 25
            },
            'file_analysis': {
                'total_files': 30,
                'total_storage_mb': 15.5,
                'avg_file_size_mb': 0.52
            }
        }
        
        # Create reports directory structure
        reports_dir = temp_dir / "reports" / "@test_channel"
        reports_dir.mkdir(parents=True)
        
        # Save report as JSON
        report_file = reports_dir / "report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f)
        
        return str(report_file)
    
    def test_generator_initialization(self, generator):
        """Test ReportDashboardGenerator initialization."""
        assert generator.reports_dir is not None
        assert generator.output_dir is not None
        assert generator.template_dir is not None
        assert generator.jinja_env is not None
    
    def test_setup_jinja2_success(self, temp_dir):
        """Test successful Jinja2 setup."""
        template_dir = temp_dir / "templates"
        template_dir.mkdir()
        
        # Create a simple template
        with open(template_dir / "test.html", "w") as f:
            f.write("<html>{{ test_var }}</html>")
        
        generator = ReportDashboardGenerator(
            str(temp_dir / "reports"),
            str(temp_dir / "dashboards"),
            str(template_dir)
        )
        
        assert generator.jinja_env is not None
    
    def test_setup_jinja2_template_dir_not_found(self, temp_dir):
        """Test Jinja2 setup with non-existent template directory."""
        with pytest.raises(FileNotFoundError):
            ReportDashboardGenerator(
                str(temp_dir / "reports"),
                str(temp_dir / "dashboards"),
                str(temp_dir / "non_existent_templates")
            )
    
    def test_generate_channel_charts(self, generator, sample_channel_report):
        """Test generating charts for a channel."""
        # Load the report data
        with open(sample_channel_report, 'r') as f:
            report_data = json.load(f)
        
        charts = generator._generate_channel_charts(report_data)
        
        assert 'time_series_chart' in charts
        assert 'media_pie_chart' in charts
        assert 'file_size_histogram' in charts
        assert 'hourly_activity' in charts
        assert 'content_analysis' in charts
        
        # Check that charts contain Plotly JSON
        for chart_name, chart_data in charts.items():
            assert 'data' in chart_data
            assert 'layout' in chart_data
            assert 'config' in chart_data
    
    def test_generate_overview_chart(self, generator):
        """Test generating overview chart."""
        channel_reports = {
            '@test_channel': {
                'channel_info': {'total_messages': 100},
                'media_analysis': {'total_media': 30}
            },
            '@another_channel': {
                'channel_info': {'total_messages': 200},
                'media_analysis': {'total_media': 60}
            }
        }
        
        chart = generator._generate_overview_chart(channel_reports)
        
        assert 'data' in chart
        assert 'layout' in chart
        assert 'config' in chart
        
        # Check that data contains both channels
        data = chart['data']
        assert len(data) == 1  # One trace
        assert len(data[0]['x']) == 2  # Two channels
        assert len(data[0]['y']) == 2  # Two message counts
    
    @pytest.mark.asyncio
    async def test_generate_channel_dashboard_success(self, generator, sample_channel_report):
        """Test successful channel dashboard generation."""
        result = generator.generate_channel_dashboard('@test_channel')
        
        assert result['status'] == 'success'
        assert result['dashboard']['html_file'] is not None
        assert result['dashboard']['output_dir'] is not None
        
        # Check if HTML file was created
        html_file = result['dashboard']['html_file']
        assert Path(html_file).exists()
        
        # Check HTML content
        with open(html_file, 'r') as f:
            content = f.read()
            assert '@test_channel Dashboard' in content
            assert 'time_series_chart' in content
            assert 'media_pie_chart' in content
    
    @pytest.mark.asyncio
    async def test_generate_channel_dashboard_report_not_found(self, generator):
        """Test channel dashboard generation with missing report."""
        result = generator.generate_channel_dashboard('@nonexistent_channel')
        
        assert result['status'] == 'failed'
        assert 'not found' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_index_dashboard_success(self, generator, sample_channel_report):
        """Test successful index dashboard generation."""
        result = generator.generate_index_dashboard()
        
        assert result['status'] == 'success'
        assert result['dashboard']['html_file'] is not None
        assert result['dashboard']['output_dir'] is not None
        
        # Check if HTML file was created
        html_file = result['dashboard']['html_file']
        assert Path(html_file).exists()
        
        # Check HTML content
        with open(html_file, 'r') as f:
            content = f.read()
            assert 'Dashboard Index' in content
            assert 'overview_chart' in content
            assert 'channel_cards' in content
    
    @pytest.mark.asyncio
    async def test_generate_all_dashboards_success(self, generator, sample_channel_report):
        """Test generating all dashboards successfully."""
        channels = ['@test_channel']
        
        results = generator.generate_all_dashboards(channels)
        
        assert len(results) == 1
        assert results['@test_channel']['status'] == 'success'
        
        # Check that both channel and index dashboards were created
        channel_result = results['@test_channel']
        assert channel_result['dashboard']['html_file'] is not None
        
        # Check index dashboard
        index_file = Path(generator.output_dir) / 'index.html'
        assert index_file.exists()
    
    def test_discover_channels(self, generator, temp_dir):
        """Test channel discovery from reports directory."""
        # Create some channel directories
        reports_dir = Path(generator.reports_dir)
        (reports_dir / "@channel1").mkdir()
        (reports_dir / "@channel2").mkdir()
        (reports_dir / "@channel3").mkdir()
        
        # Create a non-directory file
        (reports_dir / "not_a_channel.txt").touch()
        
        channels = generator._discover_channels()
        
        assert len(channels) == 3
        assert '@channel1' in channels
        assert '@channel2' in channels
        assert '@channel3' in channels
        assert 'not_a_channel.txt' not in channels
    
    def test_validate_channel_report(self, generator):
        """Test channel report validation."""
        valid_report = {
            'channel_info': {'total_messages': 100},
            'media_analysis': {'total_media': 30},
            'temporal_analysis': {'monthly_activity': {}},
            'engagement_analysis': {'total_views': 1000}
        }
        
        assert generator._validate_channel_report(valid_report) is True
        
        invalid_report = {
            'channel_info': {'total_messages': 100}
            # Missing required sections
        }
        
        assert generator._validate_channel_report(invalid_report) is False
    
    def test_create_output_directory(self, generator, temp_dir):
        """Test output directory creation."""
        new_output_dir = temp_dir / "new_dashboards"
        
        generator.output_dir = str(new_output_dir)
        generator._create_output_directory()
        
        assert new_output_dir.exists()
        assert new_output_dir.is_dir()
    
    def test_get_channel_summary(self, generator):
        """Test getting channel summary information."""
        channel_reports = {
            '@test_channel': {
                'channel_info': {
                    'total_messages': 100,
                    'channel_username': '@test_channel',
                    'date_range': {'start': '2024-01-01', 'end': '2024-01-31'}
                },
                'media_analysis': {'total_media': 30, 'total_storage_mb': 15.5},
                'engagement_analysis': {'total_views': 1000, 'total_forwards': 50}
            }
        }
        
        summary = generator._get_channel_summary(channel_reports)
        
        assert len(summary) == 1
        assert summary[0]['channel'] == '@test_channel'
        assert summary[0]['total_messages'] == 100
        assert summary[0]['total_media'] == 30
        assert summary[0]['total_storage_mb'] == 15.5
        assert summary[0]['total_views'] == 1000
        assert summary[0]['total_forwards'] == 50
