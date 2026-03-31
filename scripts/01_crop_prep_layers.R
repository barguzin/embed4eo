################################################################################
### author:    @barguzin
### created:   Mar-31-2026
### updated:   Mar-31-2026
### info:      The script crops and prepares the layers for Accra
################################################################################

library(terra)

################################################################################

# accra_bbox <- vect('~/data/aux/bbox.gpkg')
accra_bbox <- vect('~/data/aux/bbox_accra_dissolve.gpkg')
print(crs(accra_bbox))
plot(accra_bbox)

### read 10m GHSL data 
ghsl_10m_tile <- rast('~/data/GHSL_BUILD/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R9_C18/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R9_C18.tif')
ghsl_10m_tile2 <- rast('~/data/GHSL_BUILD/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R9_C19/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R9_C19.tif')

### read WSF data
wsf_data <- rast('~/data/WSF_Data/WSF2019_v1_0_4.tif')
wsf_data2 <- rast('~/data/WSF_Data/WSF2019_v1_-2_4.tif')
print(crs(wsf_data))
plot(wsf_data2)

# read embeddings data 
embed_tile <- rast('~/data/aef_accra_2019/x7dai37vmlntho3ws-0000008192-0000008192.tiff')
embed_tile2 <- rast('~/data/aef_accra_2019/xkowq9ox8kqkx6nyq-0000008192-0000000000.tiff')
plot(embed_tile[['A00']])
print(crs(embed_tile))

# use one embedding band as the exact target grid template
embed_template <- embed_tile[['A00']]

# re-project accra bbox to google reference CRS
accra_bbox <- project(accra_bbox, crs(embed_template))

# crop the template
embed_template <- crop(embed_template, accra_bbox)

# re-project WSF to google reference CRS
# IMPORTANT: use nearest-neighbor for binary raster
wsf_data2_prj <- project(wsf_data2, crs(embed_template), method = "near")

# crop WSF to accra bbox 
wsf_data_cropped2 <- crop(wsf_data2_prj, accra_bbox)
print(crs(wsf_data_cropped2))

# align WSF exactly to embedding grid
# IMPORTANT: use nearest-neighbor for binary raster
wsf_data_cropped2 <- resample(wsf_data_cropped2, embed_template, method = "near")

# binarize WSF robustly
# if your WSF is already 0/1 this does nothing harmful;
# if it uses positive values for built-up, this converts them to 1
wsf_bin <- ifel(is.na(wsf_data_cropped2), NA, ifel(wsf_data_cropped2 > 0, 1, 0))
names(wsf_bin) <- "wsf_bin"

# plot for sanity check
plot(wsf_bin)
plot(accra_bbox, add = TRUE, border = 'red', lwd = 4)

# ---------------------------------------------------------------------------
# WSF-derived features: three easy ones
# 1) binary mask
# 2) local built fraction in 5x5 window  (~50 m if pixels are ~10 m)
# 3) local built fraction in 11x11 window (~110 m if pixels are ~10 m)
# Optional 4th: distance to nearest built pixel
# ---------------------------------------------------------------------------

# local density in small neighborhood
w5 <- matrix(1, nrow = 5, ncol = 5)
wsf_dens_5 <- focal(wsf_bin, w = w5, fun = mean, na.rm = TRUE)
names(wsf_dens_5) <- "wsf_dens_5"

# local density in larger neighborhood (roughly 100 m context)
w11 <- matrix(1, nrow = 11, ncol = 11)
wsf_dens_11 <- focal(wsf_bin, w = w11, fun = mean, na.rm = TRUE)
names(wsf_dens_11) <- "wsf_dens_11"

# OPTIONAL: distance to nearest built pixel
# slightly more expensive, but very useful
wsf_built_na <- ifel(wsf_bin == 1, 1, NA)
wsf_dist <- distance(wsf_built_na)
names(wsf_dist) <- "wsf_dist"

# stack features together
wsf_feats <- c(wsf_bin, wsf_dens_5, wsf_dens_11, wsf_dist)

# quick plots
plot(wsf_feats)

# save cropped WSF binary mask
writeRaster(
  wsf_bin, 
  filename = '~/data/WSF_Data/cropped_wsf.tif', 
  overwrite = TRUE
)

# save WSF-derived features
writeRaster(
  wsf_feats,
  filename = '~/data/WSF_Data/cropped_wsf_features.tif',
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S")
)

message('Finished Processing WSF data and derived features')

# accra_bbox = project(accra_bbox, crs(embed_tile))

embed_tile_crop <- crop(embed_tile, accra_bbox)
# embed_tile_crop2 <- crop(embed_tile2, accra_bbox)

plot(embed_tile_crop[['A00']])
# plot(embed_tile_crop2[['A00']])

# embed_tiles_cropped <- merge(embed_tile_crop, embed_tile_crop2)
embed_tiles_cropped <- embed_tile_crop

# # reproject to WGS-84
# embed_tiles_cropped <- project(embed_tiles_cropped, "EPSG:4326")

# save to file 
writeRaster(embed_tiles_cropped, 
            filename = '~/data/aef_accra_2019/mosaic_accra_2019.tiff', 
            overwrite=TRUE)

message('Finished Processing Google Embeddings')
message(paste('Total number of pixels: ', ncell(embed_tiles_cropped)))

# repeat for GHSL 
ghsl <- rast('~/data/GHSL_BUILD/GHS_BUILT_S_E2019_R2023A_54009_1000_sum_trend.tif')
plot(ghsl)
print(crs(ghsl))

# accra_bbox <- project(accra_bbox, crs(ghsl))
ghsl <- project(ghsl, crs(embed_template))

ghsl_cropped <- crop(ghsl, accra_bbox, snap='out',
                     extend=T)
plot(ghsl_cropped)

### add raw GHSL pre-processing
accra_bbox2 <- project(accra_bbox, crs(ghsl_10m_tile))
# ghsl_raw <- project(ghsl_10m_tile, crs(embed_template))
# ghsl_raw2 <- project(ghsl_10m_tile2, crs(embed_template))

# ghsl_raw_cropped <- crop(ghsl_10m_tile, accra_bbox2) # these do not overlap!!
ghsl_raw_cropped2 <- crop(ghsl_10m_tile2, accra_bbox2)

ghsl_raw_cropped2 <- project(ghsl_raw_cropped2, crs(embed_template))

# ghsl_cropped <- project(ghsl_cropped, "EPSG:4326")

writeRaster(ghsl_cropped, 
            filename = '~/data/GHSL_BUILD/cropped_ghsl.tif', 
            overwrite=TRUE)

writeRaster(ghsl_raw_cropped2, 
            filename = '~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif', 
            overwrite=TRUE)

message('Finished Processing GHSL files')
