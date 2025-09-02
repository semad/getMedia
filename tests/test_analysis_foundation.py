"""
Test the analysis command foundation implementation.
"""

import pytest
import asyncio
from modules.analysis import (
    AnalysisConfig, 
    create_analysis_config, 
    run_advanced_intermediate_analysis,
    analysis_command,
    PerformanceMonitor
)


class TestAnalysisConfig:
    """Test AnalysisConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = AnalysisConfig()
        assert config.enable_file_source is True
        assert config.enable_api_source is True
        assert config.enable_diff_analysis is True
        assert config.channels == []
        assert config.verbose is False
        assert config.output_dir == "./reports/analysis"
        assert config.api_base_url == "http://localhost:8000"
        assert config.api_timeout == 30
        assert config.items_per_page == 100
        assert config.chunk_size == 10000
        assert config.memory_limit == 100000
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = AnalysisConfig()
        assert config.validate_config() is True
    
    def test_invalid_channels(self):
        """Test invalid channel names."""
        with pytest.raises(ValueError, match="Channel names must start with '@'"):
            AnalysisConfig(channels=["invalid_channel"])
    
    def test_valid_channels(self):
        """Test valid channel names."""
        config = AnalysisConfig(channels=["@test_channel", "@another_channel"])
        assert config.channels == ["@test_channel", "@another_channel"]
    
    def test_invalid_api_url(self):
        """Test invalid API URL."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            AnalysisConfig(api_base_url="invalid_url")
    
    def test_valid_api_url(self):
        """Test valid API URL."""
        config = AnalysisConfig(api_base_url="https://api.example.com")
        assert config.api_base_url == "https://api.example.com"
    
    def test_invalid_timeout(self):
        """Test invalid timeout values."""
        with pytest.raises(ValueError, match="Timeout must be between 1 and 300 seconds"):
            AnalysisConfig(api_timeout=0)
        
        with pytest.raises(ValueError, match="Timeout must be between 1 and 300 seconds"):
            AnalysisConfig(api_timeout=301)
    
    def test_valid_timeout(self):
        """Test valid timeout values."""
        config = AnalysisConfig(api_timeout=60)
        assert config.api_timeout == 60
    
    def test_invalid_items_per_page(self):
        """Test invalid items per page values."""
        with pytest.raises(ValueError, match="Items per page must be between 1 and 1000"):
            AnalysisConfig(items_per_page=0)
        
        with pytest.raises(ValueError, match="Items per page must be between 1 and 1000"):
            AnalysisConfig(items_per_page=1001)
    
    def test_valid_items_per_page(self):
        """Test valid items per page values."""
        config = AnalysisConfig(items_per_page=500)
        assert config.items_per_page == 500
    
    def test_invalid_chunk_size(self):
        """Test invalid chunk size values."""
        with pytest.raises(ValueError, match="Chunk size must be between 1000 and 50000"):
            AnalysisConfig(chunk_size=500)
        
        with pytest.raises(ValueError, match="Chunk size must be between 1000 and 50000"):
            AnalysisConfig(chunk_size=60000)
    
    def test_valid_chunk_size(self):
        """Test valid chunk size values."""
        config = AnalysisConfig(chunk_size=5000)
        assert config.chunk_size == 5000
    
    def test_invalid_memory_limit(self):
        """Test invalid memory limit values."""
        with pytest.raises(ValueError, match="Memory limit must be between 10000 and 1000000 messages"):
            AnalysisConfig(memory_limit=5000)
        
        with pytest.raises(ValueError, match="Memory limit must be between 10000 and 1000000 messages"):
            AnalysisConfig(memory_limit=2000000)
    
    def test_valid_memory_limit(self):
        """Test valid memory limit values."""
        config = AnalysisConfig(memory_limit=50000)
        assert config.memory_limit == 50000


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.start_time is None
        assert monitor.memory_start is None
    
    def test_performance_monitor_start(self):
        """Test PerformanceMonitor start method."""
        monitor = PerformanceMonitor()
        monitor.start()
        assert monitor.start_time is not None
        assert monitor.memory_start is not None
    
    def test_performance_monitor_stats_before_start(self):
        """Test PerformanceMonitor stats before start."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats()
        assert stats == {}
    
    def test_performance_monitor_stats_after_start(self):
        """Test PerformanceMonitor stats after start."""
        monitor = PerformanceMonitor()
        monitor.start()
        stats = monitor.get_stats()
        assert "elapsed_time" in stats
        assert "memory_usage_mb" in stats
        assert "memory_delta_mb" in stats
        assert isinstance(stats["elapsed_time"], float)
        assert isinstance(stats["memory_usage_mb"], float)
        assert isinstance(stats["memory_delta_mb"], float)


class TestAnalysisFunctions:
    """Test analysis functions."""
    
    def test_create_analysis_config(self):
        """Test create_analysis_config function."""
        config = create_analysis_config(
            channels=["@test_channel"],
            verbose=True,
            api_timeout=60
        )
        assert isinstance(config, AnalysisConfig)
        assert config.channels == ["@test_channel"]
        assert config.verbose is True
        assert config.api_timeout == 60
    
    @pytest.mark.asyncio
    async def test_run_advanced_intermediate_analysis(self):
        """Test run_advanced_intermediate_analysis function."""
        config = create_analysis_config(verbose=True)
        result = await run_advanced_intermediate_analysis(config)
        
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "message" in result
        assert "config" in result
        assert "timestamp" in result
        assert result["config"]["verbose"] is True
    
    def test_analysis_command(self):
        """Test analysis_command function."""
        result = analysis_command(
            channels=["@test_channel"],
            verbose=True
        )
        
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "message" in result
        assert "config" in result
        assert "timestamp" in result
        assert result["config"]["channels"] == ["@test_channel"]
        assert result["config"]["verbose"] is True


if __name__ == "__main__":
    pytest.main([__file__])
