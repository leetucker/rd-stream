# R's rdrobust 4.1.1 on every case of tests/data/cases.json, written to
# tests/data/references.json. Run from the repository root:
#   Rscript benchmarks/external/rdrobust_references.R
suppressMessages({library(arrow); library(rdrobust); library(jsonlite)})
cases <- fromJSON("tests/data/cases.json", simplifyVector = FALSE)
frames <- list()
out <- list()
for (cs in cases) {
  if (is.null(frames[[cs$data]])) frames[[cs$data]] <- as.data.frame(read_parquet(sprintf("tests/data/%s.parquet", cs$data)))
  d <- frames[[cs$data]]
  xn <- if (is.null(cs[["x"]])) "x" else cs[["x"]]
  yn <- if (is.null(cs[["y"]])) "y" else cs[["y"]]
  args <- list(y = d[[yn]], x = d[[xn]])
  for (k in c("c", "p", "q", "deriv", "kernel", "vce", "bwselect", "bwcheck", "bwrestrict",
              "stdvars", "scaleregul", "nnmatch", "rho", "scalepar", "masspoints"))
    if (!is.null(cs[[k]])) args[[k]] <- cs[[k]]
  if (!is.null(cs[["h"]])) args$h <- unlist(cs[["h"]])
  if (!is.null(cs[["b"]])) args$b <- unlist(cs[["b"]])
  if (isTRUE(cs[["cluster"]])) args$cluster <- d$cl
  if (isTRUE(cs[["weights"]])) args$weights <- d$w
  if (!is.null(cs[["fuzzy"]])) args$fuzzy <- d[[cs[["fuzzy"]]]]
  if (!is.null(cs[["sharpbw"]])) args$sharpbw <- cs[["sharpbw"]]
  if (!is.null(cs[["covs"]])) args$covs <- as.matrix(d[, unlist(cs[["covs"]]), drop = FALSE])
  rec <- list(name = cs$name)
  res <- tryCatch({
  if (isTRUE(cs[["all"]])) {
    a2 <- args[intersect(names(args), c("y", "x", "c", "p", "q", "deriv", "kernel", "vce",
          "bwcheck", "bwrestrict", "stdvars", "scaleregul", "nnmatch", "masspoints", "cluster", "weights", "covs", "fuzzy", "sharpbw"))]
    a2$all <- TRUE
    bw <- suppressWarnings(do.call(rdbwselect, a2))
    rec$bws <- lapply(seq_len(nrow(bw$bws)), function(i) c(list(selector = rownames(bw$bws)[i]), as.list(unname(bw$bws[i, ]))))
  } else {
    r <- suppressWarnings(do.call(rdrobust, args))
    rec$coef <- unname(r$coef[, 1]); rec$se <- unname(r$se[, 1])
    rec$h <- unname(r$bws[1, ]); rec$b <- unname(r$bws[2, ])
    rec$n_h <- unname(r$N_h); rec$n_b <- unname(r$N_b); rec$n <- unname(r$N)
    if (!is.null(cs[["fuzzy"]])) { rec$tau_T <- unname(unlist(r$tau_T)); rec$se_T <- unname(unlist(r$se_T)) }
  }
  rec }, error = function(e) { rec$error <- conditionMessage(e); rec },
         warning = function(w) { rec$error <- conditionMessage(w); rec })
  rec <- res
  out[[length(out) + 1]] <- rec
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "tests/data/references.json")
cat(length(out), "references\n")
