"""Expose the reusable semflow CLI under the pipeline composition root."""

from semflow.cli import app as commands

__all__ = ["commands"]
