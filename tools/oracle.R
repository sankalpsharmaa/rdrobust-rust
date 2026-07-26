#!/usr/bin/env Rscript

# JSON-lines are deliberately avoided: one request array enters on stdin and
# one result array leaves on stdout. All diagnostics and failures go to stderr.

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("tools/oracle.R requires the R package 'jsonlite'", call. = FALSE)
}
if (!requireNamespace("rdrobust", quietly = TRUE)) {
  stop("tools/oracle.R requires the R package 'rdrobust'", call. = FALSE)
}

input <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\n")
requests <- jsonlite::fromJSON(input, simplifyVector = FALSE)
if (!is.list(requests)) {
  stop("oracle request must be a JSON array", call. = FALSE)
}

vec <- function(value) {
  if (is.null(value)) NULL else as.numeric(unlist(value, use.names = FALSE))
}

scalar <- function(value, default = NULL) {
  if (is.null(value)) default else unlist(value, use.names = FALSE)[[1]]
}

logical_scalar <- function(value, default = FALSE) {
  if (is.null(value)) default else isTRUE(as.logical(scalar(value)))
}

covariates <- function(case) {
  if (is.null(case$z)) return(NULL)
  matrix(vec(case$z), ncol = as.integer(scalar(case$dz)), byrow = TRUE)
}

rdrobust_expected <- function(out, x = NULL, cluster = NULL) {
  expected <- list(
    coef = as.numeric(out$coef[, 1]),
    se = as.numeric(out$se[, 1]),
    z = as.numeric(out$z[, 1]),
    pv = as.numeric(out$pv[, 1]),
    ci_l = as.numeric(out$ci[, 1]),
    ci_u = as.numeric(out$ci[, 2]),
    bws_h = as.numeric(out$bws[1, ]),
    bws_b = as.numeric(out$bws[2, ])
  )
  if (!is.null(out$N_h)) expected$N_h <- as.numeric(out$N_h)
  if (!is.null(out$N)) expected$N <- as.numeric(out$N)
  if (!is.null(cluster) && !is.null(x)) {
    cutoff <- if (is.null(out$c)) 0 else as.numeric(out$c)
    expected$gl <- length(unique(cluster[x < cutoff]))
    expected$gr <- length(unique(cluster[x >= cutoff]))
  }
  expected
}

run_rdrobust <- function(case) {
  x <- vec(case$x)
  cluster <- vec(case$cl)
  args <- list(
    y = vec(case$y),
    x = x,
    fuzzy = vec(case$t),
    covs = covariates(case),
    covs_drop = logical_scalar(case$covs_drop_coll, FALSE),
    cluster = cluster,
    weights = vec(case$w),
    c = as.numeric(scalar(case$c, 0)),
    p = as.integer(scalar(case$p, 1)),
    q = as.integer(scalar(case$q, 2)),
    deriv = as.integer(scalar(case$deriv, 0)),
    kernel = scalar(case$kernel, "tri"),
    vce = scalar(case$vce, "nn"),
    nnmatch = as.integer(scalar(case$nnmatch, 3)),
    scaleregul = as.numeric(scalar(case$scaleregul, 1)),
    scalepar = as.numeric(scalar(case$scalepar, 1)),
    level = as.numeric(scalar(case$level, 95)),
    bwselect = scalar(case$bwselect, "mserd"),
    masspoints = scalar(case$masspoints, "off"),
    stdvars = logical_scalar(case$stdvars, FALSE),
    bwrestrict = logical_scalar(case$bwrestrict, FALSE),
    bwcheck = if (is.null(case$bwcheck)) NULL else as.integer(scalar(case$bwcheck)),
    sharpbw = logical_scalar(case$sharpbw, FALSE),
    rho = if (is.null(case$rho)) NULL else as.numeric(scalar(case$rho)),
    h = if (is.null(case$h)) NULL else as.numeric(scalar(case$h))
  )
  args <- args[!vapply(args, is.null, logical(1))]
  out <- do.call(rdrobust::rdrobust, args)
  expected <- rdrobust_expected(out)
  if (!is.null(cluster)) {
    cutoff <- as.numeric(scalar(case$c, 0))
    expected$gl <- length(unique(cluster[x < cutoff]))
    expected$gr <- length(unique(cluster[x >= cutoff]))
  }
  expected
}

run_root_rdrobust <- function(case) {
  data <- case$data
  args <- list(y = vec(data$y), x = vec(data$x))
  if (!is.null(data$fuzzy)) args$fuzzy <- vec(data$fuzzy)
  if (!is.null(data$cluster)) args$cluster <- vec(data$cluster)
  if (!is.null(data$covs)) {
    raw_covs <- data$covs
    if (is.list(raw_covs) && length(raw_covs) > 0 && is.list(raw_covs[[1]])) {
      args$covs <- do.call(rbind, lapply(raw_covs, vec))
    } else {
      args$covs <- matrix(vec(raw_covs), ncol = 1)
    }
  }
  for (name in names(case$options)) {
    args[[name]] <- scalar(case$options[[name]])
  }
  rdrobust_expected(do.call(rdrobust::rdrobust, args))
}

run_bwselect <- function(case) {
  args <- list(
    y = vec(case$y),
    x = vec(case$x),
    fuzzy = vec(case$t),
    covs = covariates(case),
    cluster = vec(case$cl),
    c = as.numeric(scalar(case$c, 0)),
    p = as.integer(scalar(case$p, 1)),
    q = as.integer(scalar(case$q, 2)),
    deriv = as.integer(scalar(case$deriv, 0)),
    kernel = scalar(case$kernel, "tri"),
    vce = scalar(case$vce, "nn"),
    nnmatch = as.integer(scalar(case$nnmatch, 3)),
    scaleregul = as.numeric(scalar(case$scaleregul, 1)),
    bwselect = scalar(case$bwselect, "mserd"),
    masspoints = "off",
    stdvars = FALSE,
    bwrestrict = FALSE
  )
  args <- args[!vapply(args, is.null, logical(1))]
  bws <- do.call(rdrobust::rdbwselect, args)$bws
  list(
    h_l = as.numeric(bws[1, 1]),
    h_r = as.numeric(bws[1, 2]),
    b_l = as.numeric(bws[1, 3]),
    b_r = as.numeric(bws[1, 4])
  )
}

run_bw <- function(case) {
  x <- vec(case$x)
  n <- length(x)
  if (identical(scalar(case$vce), "nn")) {
    runs <- rle(x)
    dups <- rep(runs$lengths, runs$lengths)
    dupsid <- sequence(runs$lengths)
  } else {
    dups <- rep(1, n)
    dupsid <- rep(1, n)
  }
  helper <- getFromNamespace("rdrobust_bw", "rdrobust")
  out <- helper(
    Y = vec(case$y),
    X = x,
    T = vec(case$t),
    Z = covariates(case),
    C = NULL,
    W = NULL,
    c = as.numeric(scalar(case$c)),
    o = as.integer(scalar(case$o)),
    nu = as.integer(scalar(case$nu)),
    o_B = as.integer(scalar(case$o_b)),
    h_V = as.numeric(scalar(case$h_v)),
    h_B = as.numeric(scalar(case$h_b)),
    scale = as.numeric(scalar(case$scale)),
    vce = scalar(case$vce),
    nnmatch = as.integer(scalar(case$nnmatch, 3)),
    kernel = scalar(case$kernel),
    dups = dups,
    dupsid = dupsid,
    covs_drop_coll = logical_scalar(case$covs_drop_coll, FALSE),
    ginv.tol = 1e-20,
    vcache = NULL
  )
  list(
    v = as.numeric(out$V),
    b = as.numeric(out$B),
    r = as.numeric(out$R),
    rate = as.numeric(out$rate)
  )
}

run_rdplot <- function(case) {
  scale_l <- as.numeric(scalar(case$scale_l, 1))
  scale_r <- as.numeric(scalar(case$scale_r, 1))
  scale <- if (scale_l == 1 && scale_r == 1) {
    NULL
  } else if (scale_l == scale_r) {
    scale_l
  } else {
    c(scale_l, scale_r)
  }
  nbins <- if (is.null(case$nbins_l) || is.null(case$nbins_r)) {
    NULL
  } else {
    c(as.integer(scalar(case$nbins_l)), as.integer(scalar(case$nbins_r)))
  }
  args <- list(
    y = vec(case$y),
    x = vec(case$x),
    c = as.numeric(scalar(case$c, 0)),
    p = as.integer(scalar(case$p, 4)),
    kernel = scalar(case$kernel, "uni"),
    binselect = scalar(case$binselect, "esmv"),
    scale = scale,
    nbins = nbins,
    weights = vec(case$w),
    masspoints = "off",
    hide = TRUE
  )
  args <- args[!vapply(args, is.null, logical(1))]
  out <- do.call(rdrobust::rdplot, args)
  bins <- out$vars_bins
  is_left <- bins$rdplot_mean_x < as.numeric(scalar(case$c, 0))
  left <- bins[is_left, , drop = FALSE]
  right <- bins[!is_left, , drop = FALSE]
  left <- left[order(left$rdplot_mean_x), , drop = FALSE]
  right <- right[order(right$rdplot_mean_x), , drop = FALSE]
  list(
    j_l = as.integer(out$J[1]),
    j_r = as.integer(out$J[2]),
    coef_l = as.numeric(out$coef[, "Left"]),
    coef_r = as.numeric(out$coef[, "Right"]),
    mean_x_l = as.numeric(left$rdplot_mean_x),
    mean_y_l = as.numeric(left$rdplot_mean_y),
    mean_x_r = as.numeric(right$rdplot_mean_x),
    mean_y_r = as.numeric(right$rdplot_mean_y)
  )
}

dispatch <- function(case) {
  kind <- scalar(case$kind)
  if (kind == "bw") return(run_bw(case))
  if (kind %in% c("bwselect", "cluster_bw")) return(run_bwselect(case))
  if (kind %in% c("rdplot", "weights_rdplot")) return(run_rdplot(case))
  if (!is.null(case$data)) return(run_root_rdrobust(case))
  if (kind %in% c("rdrobust", "perfcomp", "cluster", "prep", "rho", "weights")) {
    return(run_rdrobust(case))
  }
  stop(sprintf("unsupported oracle kind: %s", kind), call. = FALSE)
}

results <- lapply(requests, function(case) {
  expected <- dispatch(case)
  expected_keys <- unlist(case$`_expected_keys`, use.names = FALSE)
  missing <- setdiff(expected_keys, names(expected))
  if (length(missing) > 0) {
    stop(
      sprintf(
        "%s/%s oracle omitted expected fields: %s",
        scalar(case$kind),
        scalar(case$id),
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  list(
    id = scalar(case$id),
    kind = scalar(case$kind),
    expected = expected[expected_keys]
  )
})

cat(jsonlite::toJSON(results, auto_unbox = TRUE, digits = 17, null = "null"))
