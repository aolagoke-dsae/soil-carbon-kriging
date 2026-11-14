# Rule G1 (RUNBOOK, Stage 5, D7): gstat krigs the cells and the field mean
# with the parameters in the fixture. Called by src/checks/gstat.py:
#     Rscript src/checks/gstat_check.R input.json output.json
suppressPackageStartupMessages({library(gstat); library(sp); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
p <- fromJSON(args[1])
d <- data.frame(x = p$x, y = p$y, z = p$z)
coordinates(d) <- ~x + y
cells <- data.frame(x = p$cx, y = p$cy)
coordinates(cells) <- ~x + y
m <- vgm(psill = p$psill, model = p$model, range = p$range, nugget = p$nugget, kappa = p$kappa,
         anis = c(p$angle, p$ratio))
pt <- krige(z ~ 1, d, cells, model = m, debug.level = 0)
blk <- krige(z ~ 1, d, SpatialPoints(data.frame(x = 0, y = 0)), model = m,
             block = data.frame(x = p$cx, y = p$cy), debug.level = 0)
out <- list(pred = pt$var1.pred, var = pt$var1.var, block_pred = blk$var1.pred, block_var = blk$var1.var,
            versions = list(R = R.version.string, gstat = as.character(packageVersion("gstat"))))
write(toJSON(out, digits = NA, auto_unbox = TRUE), args[2])
