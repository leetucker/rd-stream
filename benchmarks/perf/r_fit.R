# One rdrobust 4.1.1 fit, timed.
#   Rscript r_fit.R data.parquet variant(x|xd) vce(nn|hc1|cr1|cr2|cr3)
# cr1..cr3 cluster on cl. Prints one line: RESULT {json}
a <- commandArgs(TRUE)
suppressMessages(library(arrow))
suppressMessages(library(rdrobust))
hwm <- function() {
  z <- grep("^VmHWM", readLines("/proc/self/status"), value = TRUE)
  as.numeric(gsub("[^0-9]", "", z)) / 1e6
}
t0 <- proc.time()[[3]]
d <- read_parquet(a[1], col_select = c(a[2], "y", "cl"))
x <- d[[a[2]]]; y <- d$y; cl <- d$cl
rm(d); invisible(gc())
t1 <- proc.time()[[3]]
vce <- a[3]
r <- if (startsWith(vce, "cr")) {
  rdrobust(y, x, cluster = cl, vce = if (vce == "cr1") "hc1" else vce)
} else rdrobust(y, x, vce = vce)
t2 <- proc.time()[[3]]
cat("RESULT ", jsonlite::toJSON(list(
  read_s = t1 - t0, fit_s = t2 - t1, peak_rss_gb = hwm(),
  coef = unname(r$coef[, 1]), se = unname(r$se[, 1]),
  h = unname(r$bws[1, ]), b = unname(r$bws[2, ])),
  digits = NA, auto_unbox = TRUE), "\n", sep = "")
