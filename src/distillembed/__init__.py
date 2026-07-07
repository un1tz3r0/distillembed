"""Distill sentence-transformer teachers into tiny static-embedding models."""

from .model import StaticModel

__version__ = "0.1.0"
__all__ = ["StaticModel", "__version__"]
