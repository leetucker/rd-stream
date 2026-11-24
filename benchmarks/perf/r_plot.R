# One rdplot (rdrobust 4.1.1) call, timed, without drawing.
#   Rscript r_plot.R data.parquet variant(x|xd)
a <- commandArgs(TRUE)
suppressMessages(library(arrow))
suppressMessages(library(rdrobust))
hwm <- function() {
  z <- grep("^VmHWM", readLines("/proc/self/status"), value = TRUE)
  as.numeric(gsub("[^0-9]", "", z)) / 1e6
}
t0 <- proc.time()[[3]]
d <- read_parquet(a[1], col_select = c(a[2], "y"))
x <- d[[a[2]]]; y <- d$y
rm(d); invisible(gc())
t1 <- proc.time()[[3]]
r <- rdplot(y, x, hide = TRUE)
t2 <- proc.time()[[3]]
cat("RESULT ", jsonlite::toJSON(list(read_s = t1 - t0, fit_s = t2 - t1, peak_rss_gb = hwm(),
  J = unname(r$J)), digits = NA, auto_unbox = TRUE), "\n", sep = "")
