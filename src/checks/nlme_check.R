# Rule L1 and L2 (RUNBOOK, Stage 2, Addendum on m4): nlme's REML at the
# parameter sets in the input, and its estimates at the large fields. Called
# by src/checks/nlme.py:
#     Rscript src/checks/nlme_check.R input.json output.json
suppressPackageStartupMessages({library(nlme); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
p <- fromJSON(args[1], simplifyVector = FALSE)
num <- function(v) as.numeric(unlist(v))
ln20 <- log(20)
l1 <- list()
for (case in p$cases) {
  d <- data.frame(z = num(case$z), x = num(case$x), y = num(case$y))
  X <- do.call(rbind, lapply(case$X, num))
  ranges <- num(p$ranges); taus <- num(p$taus)
  ll <- sapply(seq_along(ranges), function(i) {
    cs <- corExp(value = c(ranges[i] / ln20, taus[i]), form = ~ x + y, nugget = TRUE, fixed = TRUE)
    as.numeric(logLik(gls(z ~ X - 1, data = d, correlation = cs, method = "REML")))
  })
  l1[[length(l1) + 1]] <- list(site = case$site, mean = case$mean, loglik = ll)
}
l2 <- list()
for (case in p$estimate) {
  d <- data.frame(z = num(case$z), x = num(case$x), y = num(case$y))
  cs <- corExp(value = c(case$start_range / ln20, case$start_tau), form = ~ x + y, nugget = TRUE)
  f <- try(gls(z ~ 1, data = d, correlation = cs, method = "REML"), silent = TRUE)
  if (inherits(f, "try-error")) {
    l2[[length(l2) + 1]] <- list(site = case$site, ok = FALSE, message = as.character(f))
  } else {
    v <- coef(f$modelStruct$corStruct, unconstrained = FALSE)
    l2[[length(l2) + 1]] <- list(site = case$site, ok = TRUE, eff_range = unname(v["range"]) * ln20,
                                 tau = unname(v["nugget"]), sill = f$sigma^2,
                                 loglik = as.numeric(logLik(f)))
  }
}
out <- list(l1 = l1, l2 = l2, versions = list(R = R.version.string, nlme = as.character(packageVersion("nlme"))))
write(toJSON(out, digits = NA, auto_unbox = TRUE), args[2])
