"""
Unit tests for the channel reporter module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from modules.channel_reporter import ChannelReporter


class TestChannelReporter:
    """Test ChannelReporter class."""
    
    @pytest.fixture
    def reporter(self, mock_db_service):
        """Create a ChannelReporter instance for testing."""
        return ChannelReporter(mock_db_service)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = {
            'message_id': [1, 2, 3, 4, 5],
            'channel_username': ['@test_channel'] * 5,
            'text': ['Message 1', 'Message 2', 'Message 3', 'Message 4', 'Message 5'],
            'date': pd.date_range('2024-01-01', periods=5, freq='H'),
            'media_type': ['text', 'document', 'photo', 'text', 'video'],
            'file_size': [None, 1024, 512, None, 2048],
            'views': [10, 15, 20, 25, 30],
            'forwards': [1, 2, 3, 4, 5],
            'replies': [0, 1, 2, 3, 4]
        }
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_reporter_initialization(self, reporter):
        """Test ChannelReporter initialization."""
        assert reporter.db_service is not None
        assert reporter.reports_dir == "./reports/channels"
    
    @pytest.mark.asyncio
    async def test_get_channel_info_pandas(self, reporter, sample_dataframe):
        """Test getting channel information using pandas."""
        result = reporter._get_channel_info_pandas(sample_dataframe)
        
        assert result['total_messages'] == 5
        assert result['channel_username'] == '@test_channel'
        assert result['date_range']['start'] == '2024-01-01 00:00:00'
        assert result['date_range']['end'] == '2024-01-01 04:00:00'
        assert result['active_days'] == 1
    
    @pytest.mark.asyncio
    async def test_get_media_analysis_pandas(self, reporter, sample_dataframe):
        """Test media analysis using pandas."""
        result = reporter._get_media_analysis_pandas(sample_dataframe)
        
        assert result['total_media'] == 3
        assert result['media_percentage'] == 60.0
        assert result['media_breakdown']['text'] == 2
        assert result['media_breakdown']['document'] == 1
        assert result['media_breakdown']['photo'] == 1
        assert result['media_breakdown']['video'] == 1
        assert result['total_storage_mb'] > 0
    
    @pytest.mark.asyncio
    async def test_get_temporal_analysis_pandas(self, reporter, sample_dataframe):
        """Test temporal analysis using pandas."""
        result = reporter._get_temporal_analysis_pandas(sample_dataframe)
        
        assert 'monthly_activity' in result
        assert 'daily_activity' in result
        assert 'hourly_activity' in result
        assert len(result['monthly_activity']) > 0
        assert len(result['daily_activity']) > 0
        assert len(result['hourly_activity']) > 0
    
    @pytest.mark.asyncio
    async def test_get_content_analysis_pandas(self, reporter, sample_dataframe):
        """Test content analysis using pandas."""
        result = reporter._get_content_analysis_pandas(sample_dataframe)
        
        assert result['avg_text_length'] > 0
        assert result['total_text_length'] > 0
        assert result['text_length_distribution'] is not None
        assert result['common_words'] is not None
        assert len(result['common_words']) > 0
    
    @pytest.mark.asyncio
    async def test_get_engagement_analysis_pandas(self, reporter, sample_dataframe):
        """Test engagement analysis using pandas."""
        result = reporter._get_engagement_analysis_pandas(sample_dataframe)
        
        assert result['total_views'] == 100
        assert result['total_forwards'] == 15
        assert result['total_replies'] == 10
        assert result['avg_views_per_message'] == 20.0
        assert result['avg_forwards_per_message'] == 3.0
        assert result['avg_replies_per_message'] == 2.0
    
    @pytest.mark.asyncio
    async def test_get_file_analysis_pandas(self, reporter, sample_dataframe):
        """Test file analysis using pandas."""
        result = reporter._get_file_analysis_pandas(sample_dataframe)
        
        assert result['total_files'] == 3
        assert result['total_storage_mb'] > 0
        assert result['avg_file_size_mb'] > 0
        assert result['file_size_distribution'] is not None
        assert result['mime_type_breakdown'] is not None
    
    @pytest.mark.asyncio
    async def test_get_correlation_analysis_pandas(self, reporter, sample_dataframe):
        """Test correlation analysis using pandas."""
        result = reporter._get_correlation_analysis_pandas(sample_dataframe)
        
        assert 'text_length_vs_engagement' in result
        assert 'file_size_vs_engagement' in result
        assert 'time_vs_engagement' in result
        assert all(isinstance(v, dict) for v in result.values())
    
    @pytest.mark.asyncio
    async def test_create_channel_report_pandas(self, reporter, sample_dataframe):
        """Test creating a complete channel report using pandas."""
        result = reporter._create_channel_report_pandas(sample_dataframe, '@test_channel')
        
        assert result['channel_info']['total_messages'] == 5
        assert result['media_analysis']['total_media'] == 3
        assert result['temporal_analysis'] is not None
        assert result['content_analysis'] is not None
        assert result['engagement_analysis'] is not None
        assert result['file_analysis'] is not None
        assert result['correlation_analysis'] is not None
    
    @pytest.mark.asyncio
    async def test_save_channel_report_pandas(self, reporter, sample_dataframe, temp_dir):
        """Test saving a channel report using pandas."""
        import os
        
        # Set custom reports directory
        reporter.reports_dir = temp_dir
        
        # Create report
        report = reporter._create_channel_report_pandas(sample_dataframe, '@test_channel')
        
        # Save report
        result = reporter._save_channel_report_pandas(report, '@test_channel')
        
        assert result is True
        
        # Check if files were created
        channel_dir = os.path.join(temp_dir, '@test_channel')
        assert os.path.exists(channel_dir)
        
        # Check for Excel file
        excel_files = [f for f in os.listdir(channel_dir) if f.endswith('.xlsx')]
        assert len(excel_files) > 0
        
        # Check for JSON file
        json_files = [f for f in os.listdir(channel_dir) if f.endswith('.json')]
        assert len(json_files) > 0
    
    @pytest.mark.asyncio
    async def test_generate_channel_report_success(self, reporter, sample_dataframe):
        """Test successful channel report generation."""
        with patch.object(reporter, '_fetch_channel_data', return_value=sample_dataframe):
            result = await reporter.generate_channel_report('@test_channel')
            
            assert result['status'] == 'success'
            assert result['channel'] == '@test_channel'
            assert result['report_info'] is not None
    
    @pytest.mark.asyncio
    async def test_generate_channel_report_no_data(self, reporter):
        """Test channel report generation with no data."""
        with patch.object(reporter, '_fetch_channel_data', return_value=pd.DataFrame()):
            result = await reporter.generate_channel_report('@test_channel')
            
            assert result['status'] == 'failed'
            assert 'no data' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_channel_report_error(self, reporter):
        """Test channel report generation with error."""
        with patch.object(reporter, '_fetch_channel_data', side_effect=Exception("Database error")):
            result = await reporter.generate_channel_report('@test_channel')
            
            assert result['status'] == 'failed'
            assert 'database error' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_all_channel_reports(self, reporter, sample_dataframe):
        """Test generating reports for all channels."""
        channels = ['@test_channel', '@another_channel']
        
        with patch.object(reporter, '_fetch_channel_data', return_value=sample_dataframe):
            results = await reporter.generate_all_channel_reports(channels)
            
            assert len(results) == 2
            assert all(result['status'] == 'success' for result in results.values())
            assert results['@test_channel']['channel'] == '@test_channel'
            assert results['@another_channel']['channel'] == '@another_channel'
    
    def test_safe_channel_name(self, reporter):
        """Test safe channel name generation."""
        assert reporter.safe_channel_name('@test_channel') == 'test_channel'
        assert reporter.safe_channel_name('@test-channel') == 'test-channel'
        assert reporter.safe_channel_name('@test.channel') == 'test.channel'
        assert reporter.safe_channel_name('test_channel') == 'test_channel'
