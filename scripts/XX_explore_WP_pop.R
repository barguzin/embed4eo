################################################################################
### author:    @barguzin
### created:   Mar-31-2026
### updated:   Mar-31-2026
### info:      The script to identify the areas of high density
################################################################################

library(terra)

################################################################################

accra_data <- '~/data/GHSL_BUILD/GHS_BUILT_S_E2019_R2023A_54009_1000_sum_trend.tif'
accra <- rast(accra_data)
print(crs(accra))

wp_data <- '~/data/WorldPop/global_pop_2019_CN_1km_R2025A_UA_v1.tif'
wp <- rast(wp_data)
print(crs(wp))

# reproject Accra 
accra <- project(accra, crs(wp))
print(crs(accra))

# crop WP with accra
wp_crop <- crop(wp, accra)
plot(wp_crop)

# save to disk 
writeRaster(wp_crop, 
            filename = '~/data/WorldPop/cropped_accra.tif', 
            overwrite=T)
