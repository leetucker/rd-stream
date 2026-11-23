# Gate 5, R reference. Args: data.parquet c p
a <- commandArgs(TRUE)
suppressMessages({library(arrow); library(rdrobust)})
d <- read_parquet(a[1]); r <- rdrobust(d$y, d$x, c = as.numeric(a[2]), p = as.integer(a[3]))
cat("RESULT ", jsonlite::toJSON(list(coef = unname(r$coef[, 1]), se = unname(r$se[, 1]),
    h = unname(r$bws[1, ]), b = unname(r$bws[2, ]), n_h = unname(r$N_h)),
    digits = NA, auto_unbox = TRUE), "\n", sep = "")
