# One fit with R's rdrobust, for benchmarks/_harness.py: read the Parquet file
# (timed as "read", as the Python libraries' reading is), fit, and print one
# line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations.
#
#   Rscript rd.R '<spec as JSON>'     (see the benchmarks' external_command)
#
# spec$fit is one of
#   rdrobust   rdrobust::rdrobust(y, x), with cluster = cl when spec$cluster
#   rdplot     rdrobust::rdplot(y, x, hide = TRUE)
# and spec$path, spec$x (the running variable's column), spec$cluster (a
# column, or absent).

spec <- jsonlite::fromJSON(commandArgs(TRUE)[1], simplifyVector = TRUE)

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]

suppressPackageStartupMessages({library(arrow); library(rdrobust)})
baseline <- hwm_mb()
t0 <- now()
cols <- unique(c(spec$x, "y", spec$cluster))
d <- read_parquet(spec$path, col_select = all_of(cols))
x <- d[[spec$x]]
y <- d$y
cl <- if (is.null(spec$cluster)) NULL else d[[spec$cluster]]
n <- length(y)
rm(d); invisible(gc())
t1 <- now()

if (spec$fit == "rdrobust") {
  r <- if (is.null(cl)) rdrobust(y, x) else rdrobust(y, x, cluster = cl)
  coefs <- list(tau_cl = r$coef[1, 1], se_rb = r$se[3, 1], h = r$bws[1, 1])
} else if (spec$fit == "rdplot") {
  r <- rdplot(y, x, hide = TRUE)
  coefs <- list(J_left = r$J[1], J_right = r$J[2])
} else {
  stop(paste("unknown fit", spec$fit))
}
t2 <- now()

res <- list(t_read = t1 - t0, t_fit = t2 - t1, n_obs = n, converged = TRUE,
            peak_memory_mb = hwm_mb(), baseline_memory_mb = baseline, peak_disk_mb = 0,
            coefs = coefs)
cat("RESULT ", jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA, null = "null"), "\n",
    sep = "")
