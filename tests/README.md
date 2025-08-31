# Test Suite for Telegram Media Library

This directory contains comprehensive unit tests and integration tests for the Telegram Media Library.

## 🧪 Test Structure

### Unit Tests
- **`test_models.py`** - Tests for data models (ChannelConfig, RateLimitConfig)
- **`test_database_service.py`** - Tests for database interaction service
- **`test_import_processor.py`** - Tests for message import functionality
- **`test_telegram_collector.py`** - Tests for Telegram message collection
- **`test_channel_reporter.py`** - Tests for channel report generation
- **`test_dashboard_generator.py`** - Tests for dashboard generation

### Integration Tests
- **`test_integration.py`** - Complete workflow tests (collect → import → reports → dashboard)

### Test Configuration
- **`conftest.py`** - Common test fixtures and configuration
- **`__init__.py`** - Test package initialization

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# List available test files
python run_tests.py --list
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestChannelConfig

# Run specific test method
pytest tests/test_models.py::TestChannelConfig::test_channel_config_creation

# Run with coverage
pytest --cov=modules --cov-report=html
```

### Test Patterns
```bash
# Run all model tests
python run_tests.py --pattern "test_models.py"

# Run all collector tests
python run_tests.py --pattern "test_telegram_collector.py"

# Run integration tests
python run_tests.py --pattern "test_integration.py"
```

## 📋 Test Coverage

The test suite covers:

### ✅ Core Functionality
- [x] Data models and validation
- [x] Database service operations
- [x] Message import processing
- [x] Telegram message collection
- [x] Channel report generation
- [x] Dashboard generation

### ✅ Workflow Integration
- [x] Complete collect → import → reports → dashboard workflow
- [x] Data consistency across workflow steps
- [x] Error handling and edge cases
- [x] File I/O operations

### ✅ Edge Cases
- [x] Empty data handling
- [x] Malformed data handling
- [x] Missing dependencies
- [x] Network failures
- [x] Invalid configurations

## 🏗️ Test Architecture

### Fixtures
- **`temp_dir`** - Temporary directory for test files
- **`sample_messages`** - Sample Telegram message data
- **`sample_channel_config`** - Sample channel configuration
- **`sample_rate_limit_config`** - Sample rate limiting configuration
- **`mock_db_service`** - Mock database service
- **`mock_telegram_client`** - Mock Telegram client

### Mocking Strategy
- **External APIs** - Telegram API, database connections
- **File I/O** - Temporary directories and files
- **Network calls** - HTTP requests and responses
- **Time-dependent operations** - Date/time operations

### Async Testing
- Uses `pytest-asyncio` for testing async functions
- Proper event loop management
- Async fixture support

## 🔧 Test Configuration

### pytest.ini
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Coverage Configuration
- **HTML reports** - Generated in `htmlcov/` directory
- **Terminal output** - Coverage summary in console
- **Module coverage** - Focuses on `modules/` package

## 📊 Test Results

### Expected Output
```
🚀 Running tests with command: python -m pytest tests/ --tb=short --strict-markers --disable-warnings
📁 Tests directory: /path/to/tests
============================================================
======================== test session starts ========================
platform darwin -- Python 3.11.0, pytest-7.4.0, pluggy-1.2.0
rootdir: /path/to/getMedia
plugins: asyncio-0.21.1, cov-4.1.0
collected 45 items

tests/test_models.py ............                              [ 27%]
tests/test_database_service.py ................                [ 62%]
tests/test_import_processor.py .............                  [ 91%]
tests/test_telegram_collector.py .....                        [100%]

======================== 45 passed in 2.34s ========================
============================================================
✅ All tests passed!
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure you're in the right directory
cd /path/to/getMedia
python run_tests.py
```

#### Missing Dependencies
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov
```

#### Permission Issues
```bash
# Make test runner executable
chmod +x run_tests.py
```

### Debug Mode
```bash
# Run with maximum verbosity
pytest -vvv --tb=long

# Run single test with debugger
pytest tests/test_models.py::TestChannelConfig::test_channel_config_creation -s
```

## 📈 Adding New Tests

### Test File Naming
- Test files should start with `test_`
- Follow the pattern: `test_<module_name>.py`

### Test Class Naming
- Test classes should start with `Test`
- Follow the pattern: `Test<ClassName>`

### Test Method Naming
- Test methods should start with `test_`
- Use descriptive names: `test_user_creation_success`

### Example Test Structure
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_feature_works_correctly(self):
        """Test that the feature works as expected."""
        # Arrange
        input_data = "test"
        
        # Act
        result = process_feature(input_data)
        
        # Assert
        assert result == "expected_output"
```

## 🎯 Test Goals

1. **Verify Functionality** - Ensure all features work correctly
2. **Prevent Regressions** - Catch breaking changes early
3. **Document Behavior** - Tests serve as living documentation
4. **Improve Design** - Testable code is usually better designed
5. **Enable Refactoring** - Confidence to improve code structure

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
