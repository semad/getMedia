"""
Unit tests for the database service module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from modules.database_service import TelegramDBService


class TestTelegramDBService:
    """Test TelegramDBService class."""
    
    @pytest.fixture
    def db_service(self):
        """Create a TelegramDBService instance for testing."""
        return TelegramDBService("http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_db_service_initialization(self, db_service):
        """Test TelegramDBService initialization."""
        assert db_service.db_url == "http://localhost:8000"
        assert db_service.session is None
    
    @pytest.mark.asyncio
    async def test_db_service_context_manager(self, db_service):
        """Test TelegramDBService as context manager."""
        async with db_service as service:
            assert service.session is not None
            assert hasattr(service.session, 'post')
            assert hasattr(service.session, 'get')
        
        # Session should be closed after context exit
        assert db_service.session is None
    
    @pytest.mark.asyncio
    async def test_check_connection_success(self, db_service):
        """Test successful connection check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.get = AsyncMock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'status': 'ok'})
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            db_service.session = mock_session
            result = await db_service.check_connection()
            
            assert result is True
            mock_session.get.assert_called_once_with("http://localhost:8000/api/v1/health")
    
    @pytest.mark.asyncio
    async def test_check_connection_failure(self, db_service):
        """Test failed connection check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.get = AsyncMock()
            mock_response = Mock()
            mock_response.status = 500
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            db_service.session = mock_session
            result = await db_service.check_connection()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_store_message_success(self, db_service):
        """Test successful message storage."""
        message = {
            'message_id': 123,
            'channel_username': '@test_channel',
            'text': 'Test message',
            'date': '2024-01-01T10:00:00'
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.post = AsyncMock()
            mock_response = Mock()
            mock_response.status = 201
            mock_response.text = AsyncMock(return_value='{"success": true}')
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            db_service.session = mock_session
            result = await db_service.store_message(message)
            
            assert result is True
            mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_message_failure(self, db_service):
        """Test failed message storage."""
        message = {
            'message_id': 123,
            'channel_username': '@test_channel',
            'text': 'Test message',
            'date': '2024-01-01T10:00:00'
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.post = AsyncMock()
            mock_response = Mock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value='{"error": "Bad request"}')
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            db_service.session = mock_session
            result = await db_service.store_message(message)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_store_messages_bulk_success(self, db_service):
        """Test successful bulk message storage."""
        messages = [
            {'message_id': 1, 'text': 'Message 1'},
            {'message_id': 2, 'text': 'Message 2'}
        ]
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.post = AsyncMock()
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'success': True, 'imported': 2})
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            db_service.session = mock_session
            result = await db_service.store_messages_bulk(messages)
            
            assert result['success'] is True
            assert result['imported'] == 2
            mock_session.post.assert_called_once()
    
    def test_clean_message_data(self, db_service):
        """Test message data cleaning."""
        message_data = {
            'message_id': 123,
            'text': 'Test message',
            'date': '2024-01-01T10:00:00',
            'created_at': '2024-01-01T10:00:00',
            'updated_at': '2024-01-01T10:00:00'
        }
        
        cleaned = db_service.clean_message_data(message_data)
        
        assert 'created_at' not in cleaned
        assert 'updated_at' not in cleaned
        assert cleaned['message_id'] == 123
        assert cleaned['text'] == 'Test message'
        assert cleaned['date'] == '2024-01-01T10:00:00'
