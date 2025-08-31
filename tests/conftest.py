"""
Common test fixtures and configuration for Telegram Media Library tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Add the parent directory to the Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.models import ChannelConfig, RateLimitConfig
from modules.database_service import TelegramDBService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)  # Return Path object instead of string
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_channel_config():
    """Sample channel configuration for testing."""
    return ChannelConfig(
        username="@test_channel",
        enabled=True,
        max_messages_per_session=100,
        priority=1
    )


@pytest.fixture
def sample_rate_limit_config():
    """Sample rate limit configuration for testing."""
    return RateLimitConfig(
        messages_per_minute=30,
        delay_between_channels=5,
        session_cooldown=300
    )


@pytest.fixture
def sample_messages():
    """Sample Telegram messages for testing."""
    return [
        {
            'message_id': 1,
            'channel_username': '@test_channel',
            'date': '2024-01-01T10:00:00',
            'text': 'Test message 1',
            'media_type': 'text',
            'file_name': None,
            'file_size': None,
            'mime_type': None,
            'views': 10,
            'forwards': 2,
            'replies': 1
        },
        {
            'message_id': 2,
            'channel_username': '@test_channel',
            'date': '2024-01-01T11:00:00',
            'text': 'Test message 2 with media',
            'media_type': 'document',
            'file_name': 'test.pdf',
            'file_size': 1024,
            'mime_type': 'application/pdf',
            'views': 15,
            'forwards': 3,
            'replies': 2
        },
        {
            'message_id': 3,
            'channel_username': '@test_channel',
            'date': '2024-01-01T12:00:00',
            'text': 'Test message 3',
            'media_type': 'photo',
            'file_name': 'test.jpg',
            'file_size': 512,
            'mime_type': 'image/jpeg',
            'views': 20,
            'forwards': 1,
            'replies': 0
        }
    ]


@pytest.fixture
def mock_db_service():
    """Mock database service for testing."""
    mock_service = Mock(spec=TelegramDBService)
    mock_service.check_connection = AsyncMock(return_value=True)
    mock_service.store_message = AsyncMock(return_value=True)
    mock_service.store_messages_bulk = AsyncMock(return_value={'success': True})
    return mock_service


@pytest.fixture
def mock_telegram_client():
    """Mock Telegram client for testing."""
    mock_client = Mock()
    mock_client.iter_messages = AsyncMock()
    mock_client.disconnect = AsyncMock()
    return mock_client


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_json_file(temp_dir, sample_messages):
    """Create a sample JSON file for testing."""
    import json
    from datetime import datetime
    
    data = {
        'metadata': {
            'collected_at': datetime.now().isoformat(),
            'channels': ['@test_channel'],
            'total_messages': len(sample_messages),
            'data_format': 'structured_dataframe',
            'fields': list(sample_messages[0].keys()) if sample_messages else []
        },
        'messages': sample_messages
    }
    
    json_file = temp_dir / 'test_messages.json'  # temp_dir is now a Path object
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return str(json_file)
