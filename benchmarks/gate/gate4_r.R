# Gate 4, R: time one mserd fit and fixed-h fits. Args: data outcome h1,h2,...
a <- commandArgs(TRUE)
suppressMessages({library(arrow); library(rdrobust)})
d <- read_parquet(a[1], col_select = c("x", a[2])); x <- d$x; y <- d[[a[2]]]; rm(d)
hs <- as.numeric(strsplit(a[3], ",")[[1]])
t0 <- proc.time()[[3]]; r <- rdrobust(y, x); sel <- proc.time()[[3]] - t0
fx <- lapply(hs, function(h) { t0 <- proc.time()[[3]]; r <- rdrobust(y, x, h = h)
  list(h = h, s = proc.time()[[3]] - t0, coef = unname(r$coef[, 1]), se = unname(r$se[, 1])) })
cat("RESULT ", jsonlite::toJSON(list(select_s = sel, h_sel = unname(r$bws[1, 1]), fixed = fx),
    digits = NA, auto_unbox = TRUE), "\n", sep = "")
