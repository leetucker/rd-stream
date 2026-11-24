# R's rdplot (rdrobust 4.1.1) on every case of tests/data/plot_cases.json,
# written to tests/data/plot_references.json. Run from the repository root:
#   Rscript benchmarks/external/rdplot_references.R
suppressMessages({library(arrow); library(rdrobust); library(jsonlite)})
cases <- fromJSON("tests/data/plot_cases.json", simplifyVector = FALSE)
frames <- list()
out <- list()
for (cs in cases) {
  if (is.null(frames[[cs$data]])) frames[[cs$data]] <- as.data.frame(read_parquet(sprintf("tests/data/%s.parquet", cs$data)))
  d <- frames[[cs$data]]
  xn <- if (is.null(cs[["x"]])) "x" else cs[["x"]]
  yn <- if (is.null(cs[["y"]])) "y" else cs[["y"]]
  args <- list(y = d[[yn]], x = d[[xn]], hide = TRUE)
  for (k in c("c", "p", "kernel", "binselect", "masspoints", "scale"))
    if (!is.null(cs[[k]])) args[[k]] <- cs[[k]]
  for (k in c("h", "nbins", "support")) if (!is.null(cs[[k]])) args[[k]] <- unlist(cs[[k]])
  if (isTRUE(cs[["weights"]])) args$weights <- d$w
  if (!is.null(cs[["covs"]])) args$covs <- as.matrix(d[, unlist(cs[["covs"]]), drop = FALSE])
  rec <- list(name = cs$name)
  r <- tryCatch(suppressMessages(suppressWarnings(do.call(rdplot, args))),
                error = function(e) { rec$error <<- conditionMessage(e); NULL })
  if (!is.null(r)) {
    vb <- r$vars_bins
    rec$coef_left <- unname(r$coef[, 1]); rec$coef_right <- unname(r$coef[, 2])
    rec$J <- unname(r$J); rec$J_IMSE <- unname(r$J_IMSE); rec$J_MV <- unname(r$J_MV)
    rec$rscale <- unname(r$rscale); rec$bin_avg <- unname(r$bin_avg); rec$bin_med <- unname(r$bin_med)
    rec$N <- unname(r$N); rec$N_h <- unname(r$N_h); rec$binselect <- r$binselect
    for (k in names(vb)) rec[[k]] <- unname(vb[[k]])
    rec$poly_x <- r$vars_poly$rdplot_x; rec$poly_y <- r$vars_poly$rdplot_y
  }
  out[[length(out) + 1]] <- rec
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "tests/data/plot_references.json")
cat(length(out), "plot references\n")
