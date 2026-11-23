# One rdrobust 4.1.1 fit, timed. Args: data.parquet variant(x|xd) vce(nn|hc1|cluster) [profile.out]
# Prints one line: RESULT {json}
a <- commandArgs(TRUE)
suppressMessages(library(arrow))
suppressMessages(library(rdrobust))
hwm <- function() {
  z <- grep("^VmHWM", readLines("/proc/self/status"), value = TRUE)
  as.numeric(gsub("[^0-9]", "", z)) / 1024^2
}
t0 <- proc.time()[[3]]
d <- read_parquet(a[1], col_select = c(a[2], "y", "cl"))
x <- d[[a[2]]]; y <- d$y; cl <- d$cl
rm(d); invisible(gc())
t1 <- proc.time()[[3]]
prof <- length(a) >= 4
if (prof) Rprof(a[4], interval = 0.05)
r <- if (a[3] == "cluster") rdrobust(y, x, cluster = cl, vce = "hc1") else rdrobust(y, x, vce = a[3])
if (prof) Rprof(NULL)
t2 <- proc.time()[[3]]
cat("RESULT ", jsonlite::toJSON(list(
  read_s = t1 - t0, fit_s = t2 - t1, peak_gb = hwm(),
  coef = unname(r$coef[, 1]), se = unname(r$se[, 1]),
  h = unname(r$bws[1, ]), b = unname(r$bws[2, ]), n_h = unname(r$N_h)),
  digits = NA, auto_unbox = TRUE), "\n", sep = "")
