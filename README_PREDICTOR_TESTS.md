# Predictor Test Suite Documentation

This directory contains comprehensive test cases for the potassium prediction pipeline's predictor functionality.

## Test Files Overview

### 1. `test_predictor.py`
**Main predictor unit tests**
- Model loading (sklearn pipelines, AutoGluon directories)
- Single file prediction
- Batch prediction
- Error handling and edge cases
- Calibration status detection
- Feature transformation workflows

### 2. `test_autogluon_prediction.py`
**AutoGluon-specific tests**
- Legacy AutoGluon directory format
- Consistent pipeline format with AutoGluon wrapper
- GPU configuration handling
- Feature selection and dimension reduction
- Calibrated AutoGluon models
- Raw spectral mode with AutoGluon

### 3. `test_integration_prediction.py`
**End-to-end integration tests**
- Complete prediction workflows
- Data loading, cleaning, and prediction pipeline
- Wavelength standardization workflows
- Error recovery mechanisms
- Performance testing with larger batches
- Mixed success/failure scenarios

### 4. `test_utils.py`
**Test utilities and helpers**
- `SpectralDataGenerator`: Creates realistic spectral data
- `ModelMockFactory`: Creates various types of model mocks
- `ConfigFactory`: Creates test configurations
- `AssertionHelpers`: Common assertion patterns
- `TestDataSets`: Predefined test datasets
- `TestEnvironment`: Context manager for test environments

### 5. `run_predictor_tests.py`
**Test runner script**
- Convenient interface for running different test suites
- Support for coverage reporting and parallel execution
- Test suite selection (unit, autogluon, integration, all)

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-xdist

# Or with uv
uv add --dev pytest pytest-cov pytest-mock pytest-xdist
```

### Quick Start

#### Using the Test Configuration Script (Recommended)
```bash
# Quick validation with 2 models per type (fastest)
python test_config.py quick --models-per-type 2

# Run all real data tests with 3 models per type
python test_config.py tests --models-per-type 3

# Limit total models across all types (useful for very fast runs)
python test_config.py tests --max-total 10

# Run only single file prediction tests
python test_config.py tests --test-type single --models-per-type 5

# Run pytest directly with custom model counts
python test_config.py pytest --models-per-type 1

# Test specific file with limited models
python test_config.py pytest --models-per-type 2 --test-file test_real_data_prediction.py
```

#### Traditional Method
```bash
# Run all tests with default settings (5 models per type)
python run_real_data_tests.py

# Configure model counts
python run_real_data_tests.py --models-per-type 3
python run_real_data_tests.py --models-per-type 5 --max-models-total 15

# Quick validation (3 models per type)
python quick_prediction_validation.py
python quick_prediction_validation.py --models-per-type 1 --max-total 5

# Run unit tests
python run_predictor_tests.py --suite unit
```

### Direct pytest Usage
```bash
# Set environment variables to control model counts
export TEST_MODELS_PER_TYPE=3
export TEST_MAX_MODELS_TOTAL=10

# Run specific test file
pytest test_predictor.py -v

# Run real data tests with environment variables
pytest test_real_data_prediction.py -v

# Run specific test class
pytest test_predictor.py::TestPredictorModelLoading -v

# Run specific test method
pytest test_predictor.py::TestPredictorModelLoading::test_load_sklearn_model_success -v

# Run with coverage
pytest test_predictor.py --cov=src/models --cov-report=html

# Run all tests with markers
pytest -m "not slow" -v  # Exclude slow tests
pytest -m "integration" -v  # Run only integration tests
```

### Model Count Configuration

You can control how many models are tested using several methods:

1. **Command line arguments** (test_config.py and run_real_data_tests.py):
   - `--models-per-type N`: Test up to N models of each type (XGBoost, CatBoost, etc.)
   - `--max-total N`: Test at most N models total across all types

2. **Environment variables** (for pytest):
   - `TEST_MODELS_PER_TYPE=N`: Models per type
   - `TEST_MAX_MODELS_TOTAL=N`: Maximum total models

3. **Examples**:
   ```bash
   # Test only 1 model of each type (fastest)
   python test_config.py pytest --models-per-type 1

   # Test up to 3 models per type, but max 10 total
   python test_config.py tests --models-per-type 3 --max-total 10

   # Quick smoke test with minimal models
   python test_config.py quick --models-per-type 1 --max-total 5
   ```

## Test Structure and Patterns

### Test Class Organization
- **Setup classes**: Contain fixtures and common setup
- **Functionality classes**: Test specific features (loading, prediction, etc.)
- **Error handling classes**: Test error conditions and edge cases
- **Integration classes**: Test complete workflows

### Common Fixtures
```python
@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""

@pytest.fixture
def config(temp_dir):
    """Test configuration"""

@pytest.fixture
def predictor(config):
    """Predictor instance"""

@pytest.fixture
def sample_spectral_file(temp_dir):
    """Sample spectral data file"""
```

### Assertion Helpers
```python
from test_utils import AssertionHelpers

# Validate prediction results
AssertionHelpers.assert_valid_prediction(result, expected_range=(0.0, 2.0))

# Validate batch results
AssertionHelpers.assert_valid_batch_results(results_df, expected_samples=10)

# Validate model loading
AssertionHelpers.assert_model_loading(model, needs_manual_features, "Ridge")
```

## Test Categories

### Unit Tests
- **Model Loading**: Test various model formats and error conditions
- **Single Prediction**: Test individual file prediction workflows
- **Batch Prediction**: Test multi-file prediction workflows
- **Feature Handling**: Test feature transformation and selection
- **Error Handling**: Test exception handling and recovery

### AutoGluon Tests
- **Legacy Format**: Test directory-based AutoGluon models
- **Consistent Format**: Test pipeline-wrapped AutoGluon models
- **GPU Support**: Test GPU configuration and logging
- **Calibration**: Test calibrated AutoGluon predictions
- **Feature Processing**: Test AutoGluon-specific feature handling

### Integration Tests
- **End-to-End Workflows**: Complete prediction pipelines
- **Data Processing**: Integration with data cleaning and averaging
- **Configuration Integration**: Test various configuration combinations
- **Error Recovery**: Test fallback mechanisms and error handling
- **Performance**: Test with larger datasets and batch sizes

## Mock Strategy

### Spectral Data Mocking
```python
from test_utils import SpectralDataGenerator

# Create realistic spectral data
wavelengths, intensities = SpectralDataGenerator.generate_potassium_spectrum(
    wavelength_range=(400, 800),
    potassium_concentration=0.3,
    noise_level=0.1
)

# Create spectral file
file_path = SpectralDataGenerator.create_spectral_file(
    temp_dir / "sample.csv.txt",
    potassium_concentration=0.25
)
```

### Model Mocking
```python
from test_utils import ModelMockFactory

# Create sklearn pipeline mock
model = ModelMockFactory.create_sklearn_pipeline_mock(
    model_type="Ridge",
    prediction_value=0.30
)

# Create AutoGluon model directory
model_dir = ModelMockFactory.create_autogluon_model_directory(
    temp_dir,
    include_calibration=True,
    include_feature_selector=True
)
```

## Coverage Targets

The test suite aims for:
- **Unit Tests**: >95% code coverage for predictor.py
- **Integration Tests**: Coverage of all major workflows
- **Error Handling**: Coverage of all exception paths
- **Feature Branches**: Coverage of all configuration combinations

## Performance and Model Selection

### Model Testing Strategy

The test suite automatically selects the **most recent models** of each type based on modification time. This ensures you're testing with your latest and presumably best-performing models.

**Default Model Counts:**
- **Quick validation**: 3 models per type (total ~20-30 models)
- **Real data tests**: 5 models per type (total ~30-40 models)
- **Unit tests**: Use mocked models (fast)

**Recommended Configurations:**

```bash
# Ultra-fast smoke test (1-2 minutes)
python test_config.py quick --models-per-type 1 --max-total 5

# Balanced testing (5-10 minutes)
python test_config.py tests --models-per-type 3 --max-total 15

# Comprehensive testing (15-30 minutes)
python test_config.py tests --models-per-type 5

# Full validation before production (30+ minutes)
python test_config.py tests  # Uses all latest models
```

### Model Type Coverage

The tests automatically detect and categorize your models:
- **Gradient Boosting**: XGBoost, CatBoost, LightGBM
- **Tree Models**: Random Forest, Extra Trees
- **Linear**: Ridge Regression
- **Neural Networks**: Custom PyTorch models
- **AutoGluon**: Both .pkl and directory formats
- **Others**: Any additional model types

## Best Practices for Adding Tests

### 1. Use Fixtures for Common Setup
```python
@pytest.fixture
def my_test_setup(temp_dir, config):
    # Common setup logic
    return setup_object
```

### 2. Use Descriptive Test Names
```python
def test_single_prediction_with_wavelength_standardization_enabled():
    """Test single prediction when wavelength standardization is enabled"""
```

### 3. Use Test Utilities
```python
from test_utils import TestEnvironment, AssertionHelpers

def test_my_feature():
    with TestEnvironment() as (temp_dir, config):
        # Test logic
        AssertionHelpers.assert_valid_prediction(result)
```

### 4. Mock External Dependencies
```python
@patch('src.models.predictor.AUTOGLUON_AVAILABLE', True)
@patch('src.models.predictor.TabularPredictor')
def test_autogluon_functionality(mock_tabular, predictor):
    # Test logic with mocked AutoGluon
```

### 5. Test Both Success and Failure Paths
```python
def test_prediction_success(predictor):
    # Test successful prediction

def test_prediction_with_invalid_input(predictor):
    # Test error handling
    with pytest.raises(PipelineError):
        predictor.make_prediction(invalid_input, model_path)
```

## Debugging Tests

### Running Individual Tests
```bash
# Run specific test with output
pytest test_predictor.py::test_single_prediction_success -v -s

# Run with debugger
pytest test_predictor.py::test_single_prediction_success --pdb

# Run with coverage and detailed output
pytest test_predictor.py -v --cov=src/models --cov-report=term-missing
```

### Common Issues
1. **Import errors**: Ensure PYTHONPATH includes project root
2. **Missing fixtures**: Check fixture scope and dependencies
3. **Mock issues**: Verify mock paths and return values
4. **Temporary files**: Ensure proper cleanup in fixtures

## Continuous Integration

For CI/CD integration:
```yaml
# Example GitHub Actions step
- name: Run Predictor Tests
  run: |
    python run_predictor_tests.py --suite all --coverage

# Or with tox
- name: Run Tests with Tox
  run: tox -e predictor-tests
```

## Performance Considerations

- **Test Isolation**: Each test uses fresh temporary directories
- **Mock Usage**: External dependencies are mocked to avoid I/O overhead
- **Parallel Execution**: Tests can run in parallel with pytest-xdist
- **Data Generation**: Realistic but minimal test data to balance accuracy and speed

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Add appropriate docstrings
3. Use test utilities where possible
4. Include both positive and negative test cases
5. Update this documentation if adding new test files or patterns