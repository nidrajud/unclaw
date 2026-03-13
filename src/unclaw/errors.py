"""Project exception hierarchy."""


class UnclawError(Exception):
    """Base exception for the project."""


class ConfigurationError(UnclawError):
    """Raised when configuration is missing or invalid."""


class PathResolutionError(ConfigurationError):
    """Raised when a required project path cannot be resolved."""


class BootstrapError(UnclawError):
    """Raised when runtime startup preparation fails."""
