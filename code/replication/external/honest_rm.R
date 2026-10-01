# HonestDiD relative-magnitudes sensitivity (Rambachan and Roth), called from eventstudy.py.
# Input JSON: betahat, sigma, numPrePeriods, numPostPeriods, l_vec, alpha, Mbar_grid, gridPoints, bisection_tol,
#             max_doublings. Output JSON: the original CS, the CI at each Mbar, the breakdown Mbar* and the R warnings.
# Each Mbar is first run on the default theta grid; a CI open at an endpoint is rerun with the half-width doubled
# (same step) at most max_doublings times.
load_warnings <- character(0)
withCallingHandlers({
  suppressPackageStartupMessages({
    library(HonestDiD)
    library(jsonlite)
  })
}, warning = function(w) {
  load_warnings <<- c(load_warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
})
args <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1], simplifyVector = TRUE)
betahat <- as.numeric(input$betahat)
k <- length(betahat)
sigma <- matrix(as.numeric(unlist(input$sigma)), nrow = k, ncol = k, byrow = TRUE)
pre <- as.integer(input$numPrePeriods)
post <- as.integer(input$numPostPeriods)
l_vec <- matrix(as.numeric(input$l_vec), ncol = 1)
alpha <- as.numeric(input$alpha)
grid_points <- as.integer(input$gridPoints)
max_doublings <- if (is.null(input$max_doublings)) 10L else as.integer(input$max_doublings)
OPEN_MSG <- "CI is open at one of the endpoints"

post_idx <- (pre + 1):(pre + post)
sd_theta <- c(sqrt(t(l_vec) %*% sigma[post_idx, post_idx] %*% l_vec))   # HonestDiD's sdTheta
default_half_width <- 20 * sd_theta                                      # HonestDiD default bound

run_rm <- function(M, half_width = NA, n_points = grid_points) {
  warns <- character(0)
  r <- withCallingHandlers(
    if (is.na(half_width)) {
      # the plain call with HonestDiD's default grid bounds
      createSensitivityResults_relativeMagnitudes(
        betahat = betahat, sigma = sigma, numPrePeriods = pre, numPostPeriods = post,
        Mbarvec = c(M), l_vec = l_vec, alpha = alpha, gridPoints = n_points)
    } else {
      createSensitivityResults_relativeMagnitudes(
        betahat = betahat, sigma = sigma, numPrePeriods = pre, numPostPeriods = post,
        Mbarvec = c(M), l_vec = l_vec, alpha = alpha, gridPoints = n_points,
        grid.lb = -half_width, grid.ub = half_width)
    },
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  W <- if (is.na(half_width)) default_half_width else half_width
  step <- 2 * W / (n_points - 1)
  lb <- as.numeric(r$lb[1]); ub <- as.numeric(r$ub[1])
  at_bound <- (is.finite(lb) && lb <= -W + step / 2) || (is.finite(ub) && ub >= W - step / 2)
  open <- any(grepl(OPEN_MSG, warns, fixed = TRUE)) || at_bound
  list(Mbar = M, lb = lb, ub = ub, half_width = W, grid_points = n_points, step = step,
       open = open, warnings = I(unique(warns)))
}

contains0 <- function(ci) is.finite(ci[1]) && is.finite(ci[2]) && ci[1] <= 0 && ci[2] >= 0 ||
  (!is.finite(ci[1]) || !is.finite(ci[2]))

# Widen (constant step) until the CI is closed; returns the attempts after the default call.
widen <- function(M, start_half_width) {
  attempts <- list()
  W <- max(2 * default_half_width, start_half_width)
  doublings <- round(log2(W / default_half_width))
  repeat {
    n <- as.integer(round((grid_points - 1) * W / default_half_width)) + 1L
    cur <- run_rm(M, W, n)
    attempts[[length(attempts) + 1]] <- cur
    if (!cur$open || doublings >= max_doublings) break
    W <- 2 * W
    doublings <- doublings + 1
  }
  attempts
}

grid <- as.numeric(input$Mbar_grid)
default_runs <- lapply(grid, function(M) run_rm(M))
start_W <- default_half_width
grid_out <- vector("list", length(grid))
for (i in seq_along(grid)) {
  d <- default_runs[[i]]
  final <- d
  widening <- list()
  if (d$open && max_doublings > 0) {
    widening <- widen(grid[i], start_W)
    final <- widening[[length(widening)]]
    start_W <- final$half_width
  }
  grid_out[[i]] <- list(
    Mbar = grid[i], lb = final$lb, ub = final$ub,
    contains0 = contains0(c(d$lb, d$ub)),                 # decided on the default grid (rule 1)
    lb_default_grid = d$lb, ub_default_grid = d$ub,
    default_grid_open = d$open, warnings_default_grid = d$warnings,
    widened = length(widening) > 0, truncated = final$open,
    half_width = final$half_width, grid_points = final$grid_points, step = final$step,
    widening = lapply(widening, function(a) a[c("half_width", "grid_points", "lb", "ub", "open", "warnings")]))
}
first <- which(vapply(default_runs, function(d) contains0(c(d$lb, d$ub)), logical(1)))[1]
breakdown <- NULL
status <- "ok"
bisection <- list()
if (is.na(first)) {
  status <- "robust_ci_excludes_0_on_whole_grid"
  breakdown <- NA
} else if (first == 1) {
  breakdown <- grid[1]
} else {
  lo <- grid[first - 1]; hi <- grid[first]
  while (hi - lo > as.numeric(input$bisection_tol)) {
    mid <- (lo + hi) / 2
    b <- run_rm(mid)
    inside <- contains0(c(b$lb, b$ub))
    bisection[[length(bisection) + 1]] <- list(Mbar = mid, lb_default_grid = b$lb, ub_default_grid = b$ub,
                                               contains0 = inside, default_grid_open = b$open,
                                               warnings_default_grid = b$warnings)
    if (inside) hi <- mid else lo <- mid
  }
  breakdown <- hi
}
orig_warnings <- character(0)
orig <- withCallingHandlers(
  constructOriginalCS(betahat = betahat, sigma = sigma, numPrePeriods = pre,
                      numPostPeriods = post, l_vec = l_vec, alpha = alpha),
  warning = function(w) {
    orig_warnings <<- c(orig_warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
out <- list(package_version = as.character(packageVersion("HonestDiD")),
            r_version = R.version.string,
            method = "C-LF", bound = "deviation from parallel trends",
            original = list(lb = as.numeric(orig$lb), ub = as.numeric(orig$ub)),
            grid = grid_out, breakdown = breakdown, breakdown_status = status,
            bisection_tol = as.numeric(input$bisection_tol), bisection = bisection,
            sd_theta = sd_theta, default_half_width = default_half_width,
            max_doublings = max_doublings,
            warnings = list(load = I(unique(load_warnings)), original_cs = I(unique(orig_warnings))))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), args[2])
