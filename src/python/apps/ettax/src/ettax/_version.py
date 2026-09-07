# SPDX-License-Identifier: Apache-2.0
"""Installed package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ettax")
except PackageNotFoundError:
    __version__ = "0.0.0"
