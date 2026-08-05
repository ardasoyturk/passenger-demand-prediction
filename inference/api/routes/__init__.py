"""HTTP route modules for the inference API."""

from inference.api.routes import gateway, openai_compatible

__all__ = ["gateway", "openai_compatible"]
