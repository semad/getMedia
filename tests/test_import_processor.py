"""
Unit tests for the import processor module.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from modules.import_processor import (
    run_import, 
    load_messages_from_file, 
    validate_message_format, 
    check_data_quality,
    process_import_batch
)


class TestImportProcessor:
    """Test import processor functionality."""
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create a sample JSON file for testing."""
        messages = [
            {
                'message_id': 1,
                'channel_username': '@test_channel',
                'text': 'Test message 1',
                'date': '2024-01-01T10:00:00'
            },
            {
                'message_id': 2,
                'channel_username': '@test_channel',
                'text': 'Test message 2',
                'date': '2024-01-01T11:00:00'
            }
        ]
        
        data = {
            'metadata': {
                'data_format': 'structured_dataframe',
                'channels': ['@test_channel']
            },
            'messages': messages
        }
        
        json_file = Path(temp_dir) / 'test_import.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(json_file)
    
    def test_load_messages_from_file_success(self, sample_json_file):
        """Test successful message loading from file."""
        messages = load_messages_from_file(sample_json_file)
        
        assert len(messages) == 2
        assert messages[0]['message_id'] == 1
        assert messages[0]['text'] == 'Test message 1'
        assert messages[1]['message_id'] == 2
        assert messages[1]['text'] == 'Test message 2'
    
    def test_load_messages_from_file_invalid_format(self, temp_dir):
        """Test loading from file with invalid format."""
        # Create file with invalid format
        invalid_data = {
            'metadata': {
                'data_format': 'invalid_format',
                'channels': ['@test_channel']
            },
            'messages': []
        }
        
        json_file = Path(temp_dir) / 'invalid.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, ensure_ascii=False, indent=2)
        
        messages = load_messages_from_file(str(json_file))
        assert messages == []
    
    def test_load_messages_from_file_missing_messages(self, temp_dir):
        """Test loading from file with missing messages."""
        invalid_data = {
            'metadata': {
                'data_format': 'structured_dataframe',
                'channels': ['@test_channel']
            }
            # Missing 'messages' key
        }
        
        json_file = Path(temp_dir) / 'missing_messages.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, ensure_ascii=False, indent=2)
        
        messages = load_messages_from_file(str(json_file))
        assert messages == []
    
    def test_validate_message_format_valid(self):
        """Test message format validation with valid message."""
        valid_message = {
            'message_id': 1,
            'channel_username': '@test_channel',
            'text': 'Test message',
            'date': '2024-01-01T10:00:00'
        }
        
        result = validate_message_format(valid_message)
        assert result is True
    
    def test_validate_message_format_missing_required_fields(self):
        """Test message format validation with missing required fields."""
        invalid_message = {
            'message_id': 1,
            # Missing channel_username
            'text': 'Test message',
            'date': '2024-01-01T10:00:00'
        }
        
        result = validate_message_format(invalid_message)
        assert result is False
    
    def test_validate_message_format_invalid_types(self):
        """Test message format validation with invalid types."""
        invalid_message = {
            'message_id': 'not_an_integer',  # Should be int
            'channel_username': '@test_channel',
            'text': 'Test message',
            'date': '2024-01-01T10:00:00'
        }
        
        result = validate_message_format(invalid_message)
        assert result is False
    
    def test_check_data_quality_no_issues(self):
        """Test data quality check with no issues."""
        message = {
            'message_id': 1,
            'channel_username': '@test_channel',
            'text': 'Test message with reasonable length',
            'date': '2024-01-01T10:00:00',
            'file_size': 1024
        }
        
        issues = check_data_quality(message)
        assert len(issues) == 0
    
    def test_check_data_quality_text_too_short(self):
        """Test data quality check with text too short."""
        message = {
            'message_id': 1,
            'channel_username': '@test_channel',
            'text': 'Hi',  # Too short
            'date': '2024-01-01T10:00:00'
        }
        
        issues = check_data_quality(message)
        assert len(issues) > 0
        assert any('text too short' in issue.lower() for issue in issues)
    
    def test_check_data_quality_invalid_file_size(self):
        """Test data quality check with invalid file size."""
        message = {
            'message_id': 1,
            'channel_username': '@test_channel',
            'text': 'Test message',
            'date': '2024-01-01T10:00:00',
            'file_size': -1  # Invalid file size
        }
        
        issues = check_data_quality(message)
        assert len(issues) > 0
        assert any('file size' in issue.lower() for issue in issues)
    
    @pytest.mark.asyncio
    async def test_process_import_batch_success(self, mock_db_service):
        """Test successful batch processing."""
        messages = [
            {'message_id': 1, 'text': 'Message 1'},
            {'message_id': 2, 'text': 'Message 2'}
        ]
        
        mock_db_service.store_message = AsyncMock(return_value=True)
        
        success, errors, skipped = await process_import_batch(
            messages, mock_db_service, Mock(), skip_duplicates=True
        )
        
        assert success == 2
        assert errors == 0
        assert skipped == 0
        assert mock_db_service.store_message.call_count == 2
    
    @pytest.mark.asyncio
    async def test_process_import_batch_with_errors(self, mock_db_service):
        """Test batch processing with some errors."""
        messages = [
            {'message_id': 1, 'text': 'Message 1'},
            {'message_id': 2, 'text': 'Message 2'}
        ]
        
        # First message succeeds, second fails
        mock_db_service.store_message = AsyncMock(side_effect=[True, False])
        
        success, errors, skipped = await process_import_batch(
            messages, mock_db_service, Mock(), skip_duplicates=True
        )
        
        assert success == 1
        assert errors == 1
        assert skipped == 0
    
    @pytest.mark.asyncio
    async def test_run_import_validation_mode(self, sample_json_file):
        """Test import in validation mode."""
        result = await run_import(
            sample_json_file,
            "http://localhost:8000",
            validate_only=True,
            check_quality=True
        )
        
        assert result['total_messages'] == 2
        assert result['imported_count'] == 0  # No import in validation mode
        assert result['error_count'] == 0
        assert result['start_time'] is not None
        assert result['end_time'] is not None
    
    @pytest.mark.asyncio
    async def test_run_import_dry_run_mode(self, sample_json_file, mock_db_service):
        """Test import in dry run mode."""
        with patch('modules.import_processor.TelegramDBService') as mock_db_class:
            mock_db_class.return_value = mock_db_service
            
            result = await run_import(
                sample_json_file,
                "http://localhost:8000",
                dry_run=True
            )
            
            assert result['total_messages'] == 2
            assert result['imported_count'] == 2  # Counted but not actually imported
            assert result['error_count'] == 0
