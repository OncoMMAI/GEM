"""Custom exceptions for the JINA glioma embedding finetuning project."""

from typing import Optional


class BaseGliomaException(Exception):
    """Base exception class for the project."""
    
    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(BaseGliomaException):
    """Raised when there is an error in the configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        msg = f"Configuration error"
        if config_key:
            msg += f" for key '{config_key}'"
        msg += f": {message}"
        super().__init__(msg)


class DataError(BaseGliomaException):
    """Raised when there is an error in the data."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        msg = f"Data error"
        if file_path:
            msg += f" in file '{file_path}'"
        msg += f": {message}"
        super().__init__(msg)


class ModelError(BaseGliomaException):
    """Raised when there is an error in the model."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        msg = f"Model error"
        if model_name:
            msg += f" for model '{model_name}'"
        msg += f": {message}"
        super().__init__(msg)
        

class TrainingError(BaseGliomaException):
    """Raised when there is an error during training."""
    
    def __init__(self, message: str, step: Optional[int] = None):
        self.step = step
        msg = f"Training error"
        if step is not None:
            msg += f" at step {step}"
        msg += f": {message}"
        super().__init__(msg)
        

class EvaluationError(BaseGliomaException):
    """Raised when there is an error during evaluation."""
    
    def __init__(self, message: str, metric: Optional[str] = None):
        self.metric = metric
        msg = f"Evaluation error"
        if metric:
            msg += f" for metric '{metric}'"
        msg += f": {message}"
        super().__init__(msg)