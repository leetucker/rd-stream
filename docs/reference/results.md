# Results

A fit's result has hdfe-stream's reporting: `summary()`, `tidy()`,
`confint()`, `summary_json()` and `etable()`.

```{eval-rst}
.. currentmodule:: rd_stream

.. autoclass:: RDResult
   :members: with_rows, estimate, robust_ci, tidy, confint

.. autoclass:: RDMulti
   :members: estimates, with_rows

.. autoclass:: BandwidthResult

.. autoclass:: PlaceboResult
   :members: plot

.. autoclass:: RDPlot
   :members: plot

.. autoclass:: RDPlotMulti
   :members: plot
```
