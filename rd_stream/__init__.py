"""Regression discontinuity on large data.

rdrobust's local-polynomial estimates, bandwidth selectors and robust
bias-corrected inference, and rdplot's binned means, from one sort of the
running variable (or one grouping by its values) and compiled, threaded
passes over each bandwidth's observations.
"""

# first, before polars is imported: a scheduler's or container's thread and
# memory limits, which set Polars' out-of-core budget (read when it is imported)
import stream_limits as _limits

_limits.apply_scheduler_defaults()
# No OpenMP wait policy is set: the bandwidth steps' parallel passes follow one
# another closely, and run 2-5 times slower when idle threads sleep (passive).

from .api import BandwidthResult, rd_bandwidth, rd_estimate  # noqa: E402
from .placebo import PlaceboResult, rd_placebo  # noqa: E402
from .plot import RDPlot, RDPlotMulti, rd_plot  # noqa: E402
from .results import RDMulti, RDResult, etable  # noqa: E402

__version__ = "0.1.0.dev0"

__all__ = ["BandwidthResult", "PlaceboResult", "RDMulti", "RDPlot", "RDPlotMulti", "RDResult",
           "etable", "rd_bandwidth", "rd_estimate", "rd_placebo", "rd_plot"]
