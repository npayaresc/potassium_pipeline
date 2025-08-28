"""
Custom exception classes for the ML pipeline.
"""

class PipelineError(Exception):
    """Base class for exceptions in this pipeline."""
    pass

class DataValidationError(PipelineError):
    """Raised when input data validation fails."""
    pass

class ModelTrainingError(PipelineError):
    """Raised for errors during model training."""
    pass