"""
Unit tests for the telegram collector module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from modules.telegram_collector import TelegramCollector, export_messages_to_file


class TestTelegramCollector:
    """Test TelegramCollector class."""
    
    @pytest.fixture
    def collector(self, sample_rate_limit_config):
        """Create a TelegramCollector instance for testing."""
        return TelegramCollector(sample_rate_limit_config)
    
    @pytest.fixture
    def mock_message(self):
        """Create a mock Telegram message for testing."""
        message = Mock()
        message.id = 123
        message.date = None  # Will use current time
        message.text = "Test message"
        message.media = None
        message.replies = None
        message.views = None
        message.forwards = None
        return message
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector):
        """Test TelegramCollector initialization."""
        assert collector.rate_config.messages_per_minute == 30
        assert collector.client is None
        assert collector.collected_messages == []
        assert collector.stats['total_messages'] == 0
        assert collector.stats['channels_processed'] == 0
        assert collector.stats['errors'] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, collector):
        """Test successful initialization."""
        with patch('telethon.TelegramClient') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(return_value=True)
            mock_client_class.return_value = mock_client
            
            result = await collector.initialize("12345", "test_api_hash", "test_session")
            
            assert result is True
            assert collector.client is not None
            mock_client.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, collector):
        """Test failed initialization."""
        with patch('telethon.TelegramClient') as mock_client_class:
            mock_client = Mock()
            mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_class.return_value = mock_client
            
            result = await collector.initialize("test_api_id", "test_api_hash", "test_session")
            
            assert result is False
            assert collector.client is None
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, collector, mock_message):
        """Test successful message processing."""
        result = await collector.process_message(mock_message, "@test_channel")
        
        assert result is not None
        assert result['message_id'] == 123
        assert result['channel_username'] == "@test_channel"
        assert result['text'] == "Test message"
        assert result['media_type'] is None
        assert result['views'] is None
        assert result['forwards'] is None
        assert result['replies'] is None
    
    @pytest.mark.asyncio
    async def test_process_message_with_media(self, collector):
        """Test message processing with media."""
        message = Mock()
        message.id = 456
        message.date = None
        message.text = "Media message"
        message.media = Mock()
        message.media.document = Mock()
        message.media.document.attributes = [Mock(file_name="test.pdf")]
        message.media.document.size = 1024
        message.media.document.mime_type = "application/pdf"
        message.replies = None
        message.views = None
        message.forwards = None
        
        # Mock the isinstance check for MessageMediaDocument
        with patch('modules.telegram_collector.MessageMediaDocument', return_value=type(message.media)):
            result = await collector.process_message(message, "@test_channel")
            
            assert result is not None
            assert result['media_type'] == "document"
            assert result['file_name'] == "test.pdf"
            assert result['file_size'] == 1024
            assert result['mime_type'] == "application/pdf"
    
    @pytest.mark.asyncio
    async def test_process_message_with_photo(self, collector):
        """Test message processing with photo."""
        message = Mock()
        message.id = 789
        message.date = None
        message.text = "Photo message"
        message.media = Mock()
        message.media.photo = Mock()  # Add photo attribute to media
        message.replies = None
        message.views = None
        message.forwards = None
        
        result = await collector.process_message(message, "@test_channel")
        
        assert result is not None
        assert result['media_type'] == "photo"
        assert result['file_size'] == 0
    
    @pytest.mark.asyncio
    async def test_process_message_missing_id(self, collector):
        """Test message processing with missing ID."""
        message = Mock()
        message.id = None  # Missing ID
        message.text = "Test message"
        
        result = await collector.process_message(message, "@test_channel")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_collect_from_channel_success(self, collector, sample_channel_config):
        """Test successful message collection from channel."""
        # Mock the client
        collector.client = Mock()
        collector.client.iter_messages = AsyncMock()
        
        # Mock messages
        mock_message1 = Mock()
        mock_message1.id = 1
        mock_message1.date = None
        mock_message1.text = "Message 1"
        mock_message1.media = None
        mock_message1.replies = None
        mock_message1.views = None
        mock_message1.forwards = None
        
        mock_message2 = Mock()
        mock_message2.id = 2
        mock_message2.date = None
        mock_message2.text = "Message 2"
        mock_message2.media = None
        mock_message2.replies = None
        mock_message2.views = None
        mock_message2.forwards = None
        
        collector.client.iter_messages.return_value = [mock_message1, mock_message2]
        
        # Mock database service
        collector.db_service = Mock()
        collector.db_service.store_message = AsyncMock(return_value=True)
        
        result = await collector.collect_from_channel(sample_channel_config, max_messages=2)
        
        assert len(result) == 2
        assert result[0]['message_id'] == 1
        assert result[1]['message_id'] == 2
        assert collector.stats['total_messages'] == 2
        assert collector.stats['channels_processed'] == 1
    
    @pytest.mark.asyncio
    async def test_collect_from_channel_no_client(self, collector, sample_channel_config):
        """Test collection attempt without initialized client."""
        result = await collector.collect_from_channel(sample_channel_config)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_collected_messages(self, collector):
        """Test getting collected messages."""
        # Add some test messages
        test_messages = [
            {'message_id': 1, 'text': 'Test 1'},
            {'message_id': 2, 'text': 'Test 2'}
        ]
        collector.collected_messages = test_messages
        
        result = collector.get_collected_messages()
        
        assert result == test_messages
        assert result is not collector.collected_messages  # Should be a copy
    
    @pytest.mark.asyncio
    async def test_clear_collected_messages(self, collector):
        """Test clearing collected messages."""
        collector.collected_messages = [{'message_id': 1, 'text': 'Test'}]
        
        collector.clear_collected_messages()
        
        assert collector.collected_messages == []
    
    @pytest.mark.asyncio
    async def test_close(self, collector):
        """Test closing the collector."""
        collector.client = Mock()
        collector.client.disconnect = AsyncMock()
        
        await collector.close()
        
        collector.client.disconnect.assert_called_once()


class TestExportFunctions:
    """Test export-related functions."""
    
    def test_export_messages_to_file_success(self, temp_dir, sample_messages):
        """Test successful message export."""
        from modules.models import ChannelConfig
        
        channel_list = [ChannelConfig("@test_channel")]
        
        result = export_messages_to_file(
            sample_messages, 
            f"{temp_dir}/test_export", 
            channel_list, 
            Mock()
        )
        
        assert result is not None
        assert result.endswith('.json')
        
        # Verify file was created
        import json
        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data['metadata']['total_messages'] == 3
        assert data['metadata']['channels'] == ['@test_channel']
        assert data['metadata']['data_format'] == 'structured_dataframe'
        assert len(data['messages']) == 3
    
    def test_export_messages_to_file_empty_messages(self, temp_dir):
        """Test export with empty messages."""
        from modules.models import ChannelConfig
        
        channel_list = [ChannelConfig("@test_channel")]
        
        result = export_messages_to_file(
            [], 
            f"{temp_dir}/empty_export", 
            channel_list, 
            Mock()
        )
        
        assert result is not None
        assert result.endswith('.json')
        
        # Verify file was created
        import json
        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data['metadata']['total_messages'] == 0
        assert len(data['messages']) == 0
    
    def test_convert_messages_to_dataframe_format(self, sample_messages):
        """Test message conversion to DataFrame format."""
        from modules.telegram_collector import convert_messages_to_dataframe_format
        
        result = convert_messages_to_dataframe_format(sample_messages, Mock())
        
        assert len(result) == 3
        assert result[0]['message_id'] == 1
        assert result[1]['message_id'] == 2
        assert result[2]['message_id'] == 3
        assert all(isinstance(msg, dict) for msg in result)
