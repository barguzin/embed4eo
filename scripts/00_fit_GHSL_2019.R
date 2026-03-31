library(terra)

###############################################################################
# Goal:
# 1. Read 100 m GHSL BUILT-S mosaics
# 2. Fit a per-pixel linear trend using 1975-2020 only
# 3. Predict 2019 at 100 m
# 4. Aggregate to 1 km using SUM
# 5. Save final 1 km single-band GeoTIFF to $HOME/data/GHSL_BUILD/
###############################################################################

# -------------------------------
# Inputs
# -------------------------------
in_dir  <- "~/Documents/GitHub/futurepop25/data/GHS_BUILT_S_R2023A/mosaics"
out_dir <- file.path(Sys.getenv("HOME"), "data", "GHSL_BUILD")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Use historical stack only; do NOT include 2025, 2030
epochs_fit <- c(1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020)
pred_year  <- 2019

# Expected file names from your prep script
files <- file.path(
  in_dir,
  paste0("GHS_BUILT_S_E", epochs_fit, "_R2023A_54009_100_mosaic_100m.tif")
)

missing_files <- files[!file.exists(files)]
if (length(missing_files) > 0) {
  stop("Missing input files:\n", paste(missing_files, collapse = "\n"))
}

# -------------------------------
# Read stack
# -------------------------------
r <- rast(files)
names(r) <- paste0("E", epochs_fit)

# -------------------------------
# Fast per-pixel linear trend
# -------------------------------
yrs <- epochs_fit

predict_2019_from_trend <- function(v) {
  ok <- is.finite(v)
  if (sum(ok) < 2) return(NA_real_)

  x <- yrs[ok]
  y <- v[ok]

  xm <- mean(x)
  ym <- mean(y)

  denom <- sum((x - xm)^2)
  if (denom == 0) return(NA_real_)

  slope <- sum((x - xm) * (y - ym)) / denom
  intercept <- ym - slope * xm

  pred <- intercept + slope * pred_year

  # Clamp to physically meaningful range for a 100 m cell:
  # 0 to 10,000 m2 built-up surface
  pred <- max(min(pred, 10000), 0)

  return(pred)
}

terraOptions(progress = 1, memfrac = 0.8)

# Optional intermediate 100 m output
out_100m <- file.path(
  out_dir,
  "GHS_BUILT_S_E2019_R2023A_54009_100_trend_100m.tif"
)

r_2019_100m <- app(
  r,
  fun = predict_2019_from_trend,
  filename = out_100m,
  overwrite = TRUE,
  wopt = list(
    names = "built_s_2019",
    datatype = "FLT4S",
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=IF_SAFER")
  )
)

# -------------------------------
# Aggregate to 1 km
# fact = 10 because 100 m -> 1000 m
# Use SUM because values are m2 per 100 m cell
# -------------------------------
out_1km <- file.path(
  out_dir,
  "GHS_BUILT_S_E2019_R2023A_54009_1000_sum_trend.tif"
)

r_2019_1km <- aggregate(
  r_2019_100m,
  fact = 10,
  fun = sum,
  na.rm = TRUE,
  filename = out_1km,
  overwrite = TRUE,
  wopt = list(
    names = "built_s_2019",
    datatype = "FLT4S",
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=IF_SAFER")
  )
)

# -------------------------------
# Quick checks
# -------------------------------
print(r_2019_1km)
print(global(r_2019_1km, c("min", "max"), na.rm = TRUE))

plot(r_2019_1km, main = "GHSL BUILT-S 2019 (1 km, summed from 100 m)")