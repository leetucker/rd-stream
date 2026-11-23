"""Regression discontinuity on large data.

rdrobust's local-polynomial estimates, bandwidth selectors and robust
bias-corrected inference, from one sort of the running variable and compiled,
threaded passes over each bandwidth's observations.
"""

from .api import BandwidthResult, rd_bandwidth, rd_estimate
from .results import RDMulti, RDResult

__version__ = "0.1.0.dev0"

__all__ = ["BandwidthResult", "RDMulti", "RDResult", "rd_bandwidth", "rd_estimate"]
