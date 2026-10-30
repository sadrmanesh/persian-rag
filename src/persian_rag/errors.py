"""Errors raised by persian-rag."""

from __future__ import annotations


class PersianRagError(Exception):
    """Base class for operational errors."""


class ConfigError(PersianRagError):
    """The configuration is incomplete or inconsistent."""


class IndexNotFoundError(PersianRagError):
    """A saved index could not be found at the given path."""


class IndexFormatError(PersianRagError):
    """A saved index is unreadable or uses an unsupported format."""


class MissingDependencyError(PersianRagError, ImportError):
    """An optional dependency needed for this feature is not installed."""
