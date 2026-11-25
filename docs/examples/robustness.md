# Robustness checks

[`examples/robustness.py`](https://github.com/leetucker/rd-stream/blob/main/examples/robustness.py)
puts the [quickstart](quickstart.md)'s estimate through the falsification
checks of Cattaneo, Idrobo and Titiunik (2020, chapter 5): other
bandwidths, donut holes, placebo cutoffs and a placebo outcome. Each is a set
of fits on the same data, which rd-stream makes from one read.

## Other bandwidths

```python
grid = rd.rd_estimate("vote ~ margin", senate, bandwidth=[5.0, 10.0, 15.0, 20.0, 30.0])
grid.estimates()
```

```text
┌────────┬──────────┬────────────┬─────────────┬─────────────┐
│ h_left ┆ estimate ┆ robust_low ┆ robust_high ┆ n_bandwidth │
╞════════╪══════════╪════════════╪═════════════╪═════════════╡
│ 5.000  ┆ 12.271   ┆ 6.090      ┆ 20.177      ┆ 245         │
│ 10.000 ┆ 7.985    ┆ 6.595      ┆ 17.249      ┆ 451         │
│ 15.000 ┆ 7.487    ┆ 4.694      ┆ 13.477      ┆ 607         │
│ 20.000 ┆ 7.270    ┆ 4.330      ┆ 11.999      ┆ 735         │
│ 30.000 ┆ 7.279    ┆ 4.055      ┆ 10.615      ┆ 931         │
└────────┴──────────┴────────────┴─────────────┴─────────────┘
```

A list of bandwidths gives one fit each, from one sort; past the first, a
bandwidth costs well under a millisecond, whatever the size of the data. The
estimate is stable from 10 percentage points out; at 5, with 245 elections,
it is larger and noisier.

## Donut holes

The elections closest to the cutoff are the likeliest to have been
manipulated, and they weigh most in the local fit. A donut hole leaves them
out and fits again, bandwidth selection included:

```python
donuts = rd.rd_estimate("vote ~ margin", senate, donut=[0.0, 0.5, 1.0, 2.0])
```

```text
┌────────┬──────────┬────────────┬─────────────┬─────────────┬───────┐
│ h_left ┆ estimate ┆ robust_low ┆ robust_high ┆ n_bandwidth ┆ donut │
╞════════╪══════════╪════════════╪═════════════╪═════════════╪═══════╡
│ 17.754 ┆ 7.414    ┆ 4.094      ┆ 10.919      ┆ 683         ┆ 0.000 │
│ 18.209 ┆ 7.092    ┆ 3.196      ┆ 10.834      ┆ 672         ┆ 0.500 │
│ 18.354 ┆ 6.581    ┆ 2.187      ┆ 10.604      ┆ 659         ┆ 1.000 │
│ 19.439 ┆ 4.696    ┆ 0.208      ┆ 8.955       ┆ 632         ┆ 2.000 │
└────────┴──────────┴────────────┴─────────────┴─────────────┴───────┘
```

`donut=d` leaves out the elections with |margin| < d (`donut=(left, right)`
for different radii; `donut_closed=` to drop those exactly at the radius
too). The estimate shrinks as the hole widens, but stays positive, and the
fits extrapolate further across the hole.

## Placebo cutoffs

At a cutoff where nothing changes, there should be no jump. `rd_placebo`
fits at each placebo cutoff using only the elections on its side of the true
cutoff (losses for cutoffs below 0, wins above), so that the true jump cannot
leak in:

```python
placebo = rd.rd_placebo("vote ~ margin", senate, placebo_cutoffs=[-10, -5, -2.5, 2.5, 5, 10])
placebo.table
```

```text
┌─────────┬─────────┬───────┬──────────┬────────────┬─────────────┬──────────┐
│ cutoff  ┆ kind    ┆ side  ┆ estimate ┆ robust_low ┆ robust_high ┆ robust_p │
╞═════════╪═════════╪═══════╪══════════╪════════════╪═════════════╪══════════╡
│ -10.000 ┆ placebo ┆ left  ┆ -23.216  ┆ -42.861    ┆ -7.857      ┆ 0.005    │
│ -5.000  ┆ placebo ┆ left  ┆ -0.177   ┆ -12.809    ┆ 8.871       ┆ 0.722    │
│ -2.500  ┆ placebo ┆ left  ┆ 2.613    ┆ -1.839     ┆ 12.339      ┆ 0.147    │
│ 0.000   ┆ actual  ┆ null  ┆ 7.414    ┆ 4.094      ┆ 10.919      ┆ 0.000    │
│ 2.500   ┆ placebo ┆ right ┆ -4.367   ┆ -16.982    ┆ 2.164       ┆ 0.129    │
│ 5.000   ┆ placebo ┆ right ┆ 3.321    ┆ -4.719     ┆ 7.311       ┆ 0.673    │
│ 10.000  ┆ placebo ┆ right ┆ 2.915    ┆ -1.087     ┆ 11.037      ┆ 0.108    │
└─────────┴─────────┴───────┴──────────┴────────────┴─────────────┴──────────┘
```

Each fit selects its own bandwidth, and at −10 the selector picks a very
narrow one: 1.7 percentage points, with 27 and 32 elections on either side
(the table's `h_left`, `n_left` and `n_right`). The estimate there is noise;
R's and Python's rdrobust give the same numbers on that subset. Giving every
cutoff the true cutoff's bandwidth makes the fits comparable:

```python
h = fit.bandwidth[0]
common = rd.rd_placebo("vote ~ margin", senate, placebo_cutoffs=[-10, -5, -2.5, 2.5, 5, 10],
                       bandwidth=h)
common.plot(xlabel="cutoff (Democratic margin, percentage points)")
```

```text
┌─────────┬─────────┬──────────┬────────────┬─────────────┬──────────┬────────┬─────────┐
│ cutoff  ┆ kind    ┆ estimate ┆ robust_low ┆ robust_high ┆ robust_p ┆ n_left ┆ n_right │
╞═════════╪═════════╪══════════╪════════════╪═════════════╪══════════╪════════╪═════════╡
│ -10.000 ┆ placebo ┆ 0.031    ┆ -10.321    ┆ 1.972       ┆ 0.183    ┆ 217    ┆ 245     │
│ -5.000  ┆ placebo ┆ 2.099    ┆ -9.159     ┆ 3.470       ┆ 0.377    ┆ 288    ┆ 128     │
│ -2.500  ┆ placebo ┆ 2.460    ┆ -1.797     ┆ 10.801      ┆ 0.161    ┆ 328    ┆ 63      │
│ 0.000   ┆ actual  ┆ 7.414    ┆ 4.274      ┆ 12.368      ┆ 0.000    ┆ 360    ┆ 323     │
│ 2.500   ┆ placebo ┆ -2.168   ┆ -13.774    ┆ 4.629       ┆ 0.330    ┆ 57     ┆ 291     │
│ 5.000   ┆ placebo ┆ 2.234    ┆ -2.772     ┆ 8.024       ┆ 0.340    ┆ 117    ┆ 258     │
│ 10.000  ┆ placebo ┆ -1.545   ┆ -5.882     ┆ 2.994       ┆ 0.524    ┆ 206    ┆ 227     │
└─────────┴─────────┴──────────┴────────────┴─────────────┴──────────┴────────┴─────────┘
```

```{image} ../figures/robustness_placebo.svg
:class: benchmark
:alt: Estimates and robust intervals at placebo cutoffs from -10 to 10, all including zero, and at the true cutoff, above zero
```

No placebo cutoff shows a jump. The true cutoff's interval differs from the
quickstart's because a given bandwidth sets the bias bandwidth to the same
value (as rdrobust does), where `mserd` selects a wider one.

## A placebo outcome

The vote share in the previous election for the seat was decided before the
election that sets the margin, so it should not jump at 0:

```python
lag = rd.rd_estimate("vote_lag1 ~ margin", senate)
```

```text
previous vote share: 2.459 (robust interval [-1.912, 7.707])
```

It does not: the interval includes zero.
