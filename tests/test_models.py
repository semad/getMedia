"""
Unit tests for the models module.
"""

import pytest
from modules.models import ChannelConfig, RateLimitConfig


class TestChannelConfig:
    """Test ChannelConfig class."""
    
    def test_channel_config_creation(self):
        """Test creating a ChannelConfig instance."""
        config = ChannelConfig(username="@test_channel")
        
        assert config.username == "@test_channel"
        assert config.enabled is True
        assert config.max_messages_per_session == 100
        assert config.priority == 1
    
    def test_channel_config_custom_values(self):
        """Test creating a ChannelConfig with custom values."""
        config = ChannelConfig(
            username="@custom_channel",
            enabled=False,
            max_messages_per_session=500,
            priority=5
        )
        
        assert config.username == "@custom_channel"
        assert config.enabled is False
        assert config.max_messages_per_session == 500
        assert config.priority == 5
    
    def test_channel_config_equality(self):
        """Test ChannelConfig equality."""
        config1 = ChannelConfig("@test_channel")
        config2 = ChannelConfig("@test_channel")
        config3 = ChannelConfig("@different_channel")
        
        assert config1 == config2
        assert config1 != config3


class TestRateLimitConfig:
    """Test RateLimitConfig class."""
    
    def test_rate_limit_config_creation(self):
        """Test creating a RateLimitConfig instance."""
        config = RateLimitConfig()
        
        assert config.messages_per_minute == 30
        assert config.delay_between_channels == 5
        assert config.session_cooldown == 300
    
    def test_rate_limit_config_custom_values(self):
        """Test creating a RateLimitConfig with custom values."""
        config = RateLimitConfig(
            messages_per_minute=60,
            delay_between_channels=10,
            session_cooldown=600
        )
        
        assert config.messages_per_minute == 60
        assert config.delay_between_channels == 10
        assert config.session_cooldown == 600
    
    def test_rate_limit_config_equality(self):
        """Test RateLimitConfig equality."""
        config1 = RateLimitConfig()
        config2 = RateLimitConfig()
        config3 = RateLimitConfig(messages_per_minute=60)
        
        assert config1 == config2
        assert config1 != config3
